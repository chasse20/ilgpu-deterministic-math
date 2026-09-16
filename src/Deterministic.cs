using System;
using ILGPU;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace DeterministicGPUMath
{
	// IEEE-754
	public static class Deterministic
	{
		private const int EXPONENT_SHIFT = 23;
		private const int EXPONENT_BITS = 8;
		private const int EXPONENT_MASK = 0xFF;
		private const int EXPONENT_BIAS = 127;
		private const int MANTISSA_BITS = 23;
		private const int MANTISSA_MASK = 0x7FFFFF;
		private const int IMPLICIT_ONE_BIT = 1 << MANTISSA_BITS;
		private const int EXPONENT_ALL_ONES = EXPONENT_MASK;
		private const int Q3232_FRACTION_BITS = 32;
		private const long Q3232_ONE = 1L << Q3232_FRACTION_BITS;
		private const int Q131_FRACTION_BITS = 31;
		private const uint Q131_ONE = 1u << Q131_FRACTION_BITS;
		private const int Q131_TO_MANT23_SHIFT = Q131_FRACTION_BITS - MANTISSA_BITS;
		private const uint DISCARDED_32_HALF = 0x80000000u;
		private const long LN2_Q3232 = 2977044472L; // round(ln(2) * 2^32)
		private const long LOG2E_Q3232 = 6196328019L; // round(log2(e) * 2^32)
		private const int CANONICAL_NAN_BITS = 0x7FC00000;

		// Kernel-safe bit-cast helpers
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static bool GetIsFinite( float tValue )
		{
			int tempBits = BitConverter.SingleToInt32Bits( tValue );
			int tempExp = ( tempBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;

			return tempExp != EXPONENT_ALL_ONES;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static bool GetIsFiniteBits( int tBits )
		{
			int tempExp = ( tBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;
			return tempExp != EXPONENT_ALL_ONES;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static bool GetIsIntegerBits( int tBits )
		{
			int tempExp = ( tBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;

			if ( tempExp == 0 || tempExp == EXPONENT_ALL_ONES )
			{
				return false;
			}

			int tempE = tempExp - EXPONENT_BIAS;

			if ( tempE < 0 )
			{
				return ( tBits & 0x7FFFFFFF ) == 0;
			}
			else if ( tempE >= MANTISSA_BITS )
			{
				return true;
			}

			int tempMant = tBits & MANTISSA_MASK;
			int tempFracMask = ( 1 << ( MANTISSA_BITS - tempE ) ) - 1;
			return ( tempMant & tempFracMask ) == 0;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static bool GetIsOddIntegerBits( int tBits )
		{
			int tempExp = ( tBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;

			if ( tempExp == 0 || tempExp == EXPONENT_ALL_ONES )
			{
				return false;
			}

			int tempE = tempExp - EXPONENT_BIAS;

			if ( tempE < 0 )
			{
				return false;
			}

			if ( tempE >= ( MANTISSA_BITS + 1 ) )
			{
				return false;
			}

			int tempMant = ( tBits & MANTISSA_MASK ) | IMPLICIT_ONE_BIT;
			int tempBit = MANTISSA_BITS - tempE;
			return ( ( tempMant >> tempBit ) & 1 ) != 0;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static long GetFloatBitsToQ3232( int tBits )
		{
			int tempSign = tBits >> 31;
			int tempExp = ( tBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;
			int tempMant = tBits & MANTISSA_MASK;

			if ( tempExp == 0 )
			{
				return 0;
			}

			tempMant |= IMPLICIT_ONE_BIT;

			int tempE = tempExp - EXPONENT_BIAS - MANTISSA_BITS;
			int tempShift = tempE + Q3232_FRACTION_BITS;
			ulong tempAbs = ( tempShift >= 0 ? ( (ulong)tempMant << tempShift ) : (ulong)GetShiftRightRoundToEvenS64( tempMant, -tempShift ) );

			if ( tempAbs > long.MaxValue )
			{
				return tempSign != 0 ? -long.MaxValue : long.MaxValue;
			}

			long tempV = (long)tempAbs;

			if ( tempSign != 0 )
			{
				tempV = -tempV;
			}

			return tempV;
		}

		// Deterministic LUT generation
		public static (long[] log2TableQ3232, uint[] exp2TableQ131) GetTables( int tIndexBits, ParallelOptions tParallelOptions )
		{
			if ( tIndexBits <= 0 || tIndexBits > 20 )
			{
				throw new ArgumentOutOfRangeException( nameof( tIndexBits ), "IndexBits must be in [1, 20]." );
			}

			int tempTableLength = ( 1 << tIndexBits ) + 1;
			long[] tempLog2Table = null;
			uint[] tempExp2Table = null;

			Parallel.Invoke
			(
				tParallelOptions,
				() => { tempLog2Table = GetLog2TableQ3232( tempTableLength, tParallelOptions ); },
				() => { tempExp2Table = GetExp2TableQ131( tempTableLength, tParallelOptions ); }
			);

			return (tempLog2Table, tempExp2Table);
		}

		public static long[] GetLog2TableQ3232( int tTableLength, ParallelOptions tParallelOptions )
		{
			if ( tTableLength <= 2 )
			{
				throw new ArgumentOutOfRangeException( nameof( tTableLength ), "TableLength must be > 2." );
			}

			int tempN = tTableLength - 1;
			long[] tempTable = new long[ tTableLength ];

			// Log2(1 + i/N) in signed Q32.32
			Parallel.For
			(
				0,
				tTableLength,
				tParallelOptions,
				tIndex =>
				{
					double tempM = 1.0 + ( tIndex / (double)tempN );
					double tempV = Math.Log( tempM, 2.0 );
					tempTable[ tIndex ] = (long)Math.Round( tempV * Q3232_ONE, MidpointRounding.ToEven );
				}
			);

			return tempTable;
		}

		public static uint[] GetExp2TableQ131( int tTableLength, ParallelOptions tParallelOptions )
		{
			if ( tTableLength <= 2 )
			{
				throw new ArgumentOutOfRangeException( nameof( tTableLength ), "TableLength must be > 2." );
			}

			int tempN = tTableLength - 1;
			uint[] tempTable = new uint[ tTableLength ];

			// 2^(i/N) in unsigned Q1.31
			Parallel.For
			(
				0,
				tTableLength,
				tParallelOptions,
				tIndex =>
				{
					double tempF = tIndex / (double)tempN;
					double tempV = Math.Pow( 2.0, tempF );
					long tempQ = (long)Math.Round( tempV * Q131_ONE, MidpointRounding.ToEven );

					if ( tempQ < 0 )
					{
						tempQ = 0;
					}

					if ( tempQ > uint.MaxValue )
					{
						tempQ = uint.MaxValue;
					}

					tempTable[ tIndex ] = (uint)tempQ;
				}
			);

			return tempTable;
		}

		// Pow
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetPow( float tX, float tY, long[] tLog2TableQ3232, uint[] tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );
			int tempYBits = BitConverter.SingleToInt32Bits( tY );

			if ( !GetIsFiniteBits( tempXBits ) || !GetIsFiniteBits( tempYBits ) )
			{
				if ( float.IsNaN( tX ) || float.IsNaN( tY ) )
				{
					return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
				}
				else if ( tY == 0.0f )
				{
					return 1.0f;
				}
				else if ( float.IsInfinity( tX ) )
				{
					return ( tY > 0.0f ) ? float.PositiveInfinity : 0.0f;
				}

				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}
			else if ( tY == 0.0f || tX == 1.0f )
			{
				return 1.0f;
			}
			else if ( tX == 0.0f )
			{
				return ( tY > 0.0f ) ? 0.0f : float.PositiveInfinity;
			}

			bool tempIsNegativeBase = tempXBits < 0;

			if ( tempIsNegativeBase )
			{
				if ( !GetIsIntegerBits( tempYBits ) )
				{
					return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
				}

				tempXBits &= int.MaxValue;
			}

			// Positive subnormals are normalized deterministically inside GetLog2DetQ3232FromBits
			long tempLog2XQ3232 = GetLog2DetQ3232FromBits( tempXBits, tLog2TableQ3232, tIndexBits );
			long tempYQ3232 = GetFloatBitsToQ3232( tempYBits );
			long tempZQ3232 = GetMulQ3232( tempLog2XQ3232, tempYQ3232 );
			bool tempIsNegativeExp = tempZQ3232 < 0;

			if ( tempIsNegativeExp )
			{
				tempZQ3232 = -tempZQ3232;
			}

			float tempPowPos = GetExp2DetFromQ3232( tempZQ3232, tExp2TableQ131, tIndexBits );
			float tempPow = tempIsNegativeExp ? GetReciprocalDet( tempPowPos ) : tempPowPos;

			if ( tempIsNegativeBase && GetIsOddIntegerBits( tempYBits ) )
			{
				tempPow = -tempPow;
			}

			return tempPow;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetPow( float tX, float tY, ArrayView<long> tLog2TableQ3232, ArrayView<uint> tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );
			int tempYBits = BitConverter.SingleToInt32Bits( tY );

			if ( !GetIsFiniteBits( tempXBits ) || !GetIsFiniteBits( tempYBits ) )
			{
				if ( float.IsNaN( tX ) || float.IsNaN( tY ) )
				{
					return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
				}
				else if ( tY == 0.0f )
				{
					return 1.0f;
				}
				else if ( float.IsInfinity( tX ) )
				{
					return ( tY > 0.0f ) ? float.PositiveInfinity : 0.0f;
				}

				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}
			else if ( tY == 0.0f || tX == 1.0f )
			{
				return 1.0f;
			}
			else if ( tX == 0.0f )
			{
				return ( tY > 0.0f ) ? 0.0f : float.PositiveInfinity;
			}

			bool tempIsNegativeBase = tempXBits < 0;

			if ( tempXBits < 0 )
			{
				if ( !GetIsIntegerBits( tempYBits ) )
				{
					return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
				}

				tempXBits &= int.MaxValue;
			}

			// Positive subnormals are normalized deterministically inside GetLog2DetQ3232FromBits
			long tempLog2XQ3232 = GetLog2DetQ3232FromBits( tempXBits, tLog2TableQ3232, tIndexBits );
			long tempYQ3232 = GetFloatBitsToQ3232( tempYBits );
			long tempZQ3232 = GetMulQ3232( tempLog2XQ3232, tempYQ3232 );
			bool tempIsNegativeExp = tempZQ3232 < 0;

			if ( tempIsNegativeExp )
			{
				tempZQ3232 = -tempZQ3232;
			}

			float tempPowPos = GetExp2DetFromQ3232( tempZQ3232, tExp2TableQ131, tIndexBits );
			float tempPow = tempIsNegativeExp ? GetReciprocalDet( tempPowPos ) : tempPowPos;

			if ( tempIsNegativeBase && GetIsOddIntegerBits( tempYBits ) )
			{
				tempPow = -tempPow;
			}

			return tempPow;
		}

		// Natural log, ln(x), using only deterministic integer/table math after IEEE-754 edge handling.
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetLog( float tX, long[] tLog2TableQ3232, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );

			float tempSpecial = GetLogSpecialResultFromBits( tempXBits, out bool tempHasSpecialResult );

			if ( tempHasSpecialResult )
			{
				return tempSpecial;
			}

			long tempLog2Q3232 = GetLog2DetQ3232FromBits( tempXBits, tLog2TableQ3232, tIndexBits );
			long tempLnQ3232 = GetMulQ3232( tempLog2Q3232, LN2_Q3232 );

			return GetQ3232ToFloat( tempLnQ3232 );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetLog( float tX, ArrayView<long> tLog2TableQ3232, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );

			float tempSpecial = GetLogSpecialResultFromBits( tempXBits, out bool tempHasSpecialResult );

			if ( tempHasSpecialResult )
			{
				return tempSpecial;
			}

			long tempLog2Q3232 = GetLog2DetQ3232FromBits( tempXBits, tLog2TableQ3232, tIndexBits );
			long tempLnQ3232 = GetMulQ3232( tempLog2Q3232, LN2_Q3232 );

			return GetQ3232ToFloat( tempLnQ3232 );
		}

		// Base-2 log. This exposes the same deterministic core used by Pow and Log.
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetLog2( float tX, long[] tLog2TableQ3232, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );

			float tempSpecial = GetLogSpecialResultFromBits( tempXBits, out bool tempHasSpecialResult );

			if ( tempHasSpecialResult )
			{
				return tempSpecial;
			}

			long tempLog2Q3232 = GetLog2DetQ3232FromBits( tempXBits, tLog2TableQ3232, tIndexBits );
			return GetQ3232ToFloat( tempLog2Q3232 );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetLog2( float tX, ArrayView<long> tLog2TableQ3232, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );

			float tempSpecial = GetLogSpecialResultFromBits( tempXBits, out bool tempHasSpecialResult );

			if ( tempHasSpecialResult )
			{
				return tempSpecial;
			}

			long tempLog2Q3232 = GetLog2DetQ3232FromBits( tempXBits, tLog2TableQ3232, tIndexBits );
			return GetQ3232ToFloat( tempLog2Q3232 );
		}

		// Sqrt. Uses deterministic Pow core after IEEE-754 edge handling.
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetSqrt( float tX, long[] tLog2TableQ3232, uint[] tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );

			float tempSpecial = GetSqrtSpecialResultFromBits( tempXBits, out bool tempHasSpecialResult );

			if ( tempHasSpecialResult )
			{
				return tempSpecial;
			}

			return GetPow( tX, 0.5f, tLog2TableQ3232, tExp2TableQ131, tIndexBits );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetSqrt( float tX, ArrayView<long> tLog2TableQ3232, ArrayView<uint> tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );

			float tempSpecial = GetSqrtSpecialResultFromBits( tempXBits, out bool tempHasSpecialResult );

			if ( tempHasSpecialResult )
			{
				return tempSpecial;
			}

			return GetPow( tX, 0.5f, tLog2TableQ3232, tExp2TableQ131, tIndexBits );
		}

		// Natural exponential: e^x
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetExp( float tX, uint[] tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );
			int tempAbsBits = tempXBits & 0x7FFFFFFF;

			if ( tempAbsBits > 0x7F800000 )
			{
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			if ( tempXBits == 0x7F800000 )
			{
				return float.PositiveInfinity;
			}

			if ( tempXBits == unchecked( (int)0xFF800000) )
			{
				return 0.0f;
			}

			long tempXQ3232 = GetFloatBitsToQ3232( tempXBits );
			long tempLog2ExponentQ3232 = GetMulQ3232( tempXQ3232, LOG2E_Q3232 );

			return GetExp2DetFromSignedQ3232( tempLog2ExponentQ3232, tExp2TableQ131, tIndexBits );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetExp( float tX, ArrayView<uint> tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );
			int tempAbsBits = tempXBits & 0x7FFFFFFF;

			if ( tempAbsBits > 0x7F800000 )
			{
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			if ( tempXBits == 0x7F800000 )
			{
				return float.PositiveInfinity;
			}

			if ( tempXBits == unchecked( (int)0xFF800000) )
			{
				return 0.0f;
			}

			long tempXQ3232 = GetFloatBitsToQ3232( tempXBits );
			long tempLog2ExponentQ3232 = GetMulQ3232( tempXQ3232, LOG2E_Q3232 );

			return GetExp2DetFromSignedQ3232( tempLog2ExponentQ3232, tExp2TableQ131, tIndexBits );
		}

		// Base-2 exponential: 2^x
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetExp2( float tX, uint[] tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );
			int tempAbsBits = tempXBits & 0x7FFFFFFF;

			if ( tempAbsBits > 0x7F800000 )
			{
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			if ( tempXBits == 0x7F800000 )
			{
				return float.PositiveInfinity;
			}

			if ( tempXBits == unchecked( (int)0xFF800000) )
			{
				return 0.0f;
			}

			long tempXQ3232 = GetFloatBitsToQ3232( tempXBits );

			return GetExp2DetFromSignedQ3232( tempXQ3232, tExp2TableQ131, tIndexBits );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float GetExp2( float tX, ArrayView<uint> tExp2TableQ131, int tIndexBits )
		{
			int tempXBits = BitConverter.SingleToInt32Bits( tX );
			int tempAbsBits = tempXBits & 0x7FFFFFFF;

			if ( tempAbsBits > 0x7F800000 )
			{
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			if ( tempXBits == 0x7F800000 )
			{
				return float.PositiveInfinity;
			}

			if ( tempXBits == unchecked( (int)0xFF800000) )
			{
				return 0.0f;
			}

			long tempXQ3232 = GetFloatBitsToQ3232( tempXBits );

			return GetExp2DetFromSignedQ3232( tempXQ3232, tExp2TableQ131, tIndexBits );
		}

		// Convert Q32.32 from long
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetQ3232ToFloat( long tValueQ3232 )
		{
			if ( tValueQ3232 == 0 )
			{
				return 0.0f;
			}

			bool tempIsNegative = tValueQ3232 < 0;
			ulong tempAbs = (ulong)( tempIsNegative ? -tValueQ3232 : tValueQ3232 );

			int tempHighestBit = 63;

			while ( tempHighestBit > 0 && ( ( tempAbs & ( 1ul << tempHighestBit ) ) == 0ul ) )
			{
				--tempHighestBit;
			}

			int tempExponent = tempHighestBit - Q3232_FRACTION_BITS;
			int tempOutExp = tempExponent + EXPONENT_BIAS;

			if ( tempOutExp <= 0 )
			{
				return tempIsNegative ? -0.0f : 0.0f;
			}

			if ( tempOutExp >= EXPONENT_ALL_ONES )
			{
				return tempIsNegative ? float.NegativeInfinity : float.PositiveInfinity;
			}

			ulong tempMant24;

			if ( tempHighestBit > MANTISSA_BITS )
			{
				int tempShift = tempHighestBit - MANTISSA_BITS;
				ulong tempKept = tempAbs >> tempShift;
				ulong tempDiscarded = tempAbs & ( ( 1ul << tempShift ) - 1ul );
				ulong tempHalf = 1ul << ( tempShift - 1 );

				if ( tempDiscarded > tempHalf || ( tempDiscarded == tempHalf && ( tempKept & 1ul ) != 0ul ) )
				{
					++tempKept;
				}

				tempMant24 = tempKept;

				if ( tempMant24 >= ( 1ul << ( MANTISSA_BITS + 1 ) ) )
				{
					tempMant24 >>= 1;
					++tempOutExp;

					if ( tempOutExp >= EXPONENT_ALL_ONES )
					{
						return tempIsNegative ? float.NegativeInfinity : float.PositiveInfinity;
					}
				}
			}
			else
			{
				tempMant24 = tempAbs << ( MANTISSA_BITS - tempHighestBit );
			}

			int tempMant = (int)( tempMant24 & MANTISSA_MASK );
			int tempSign = tempIsNegative ? int.MinValue : 0;
			int tempBits = tempSign | ( tempOutExp << EXPONENT_SHIFT ) | tempMant;

			return BitConverter.Int32BitsToSingle( tempBits );
		}

		// Log/sqrt special cases are handled from raw bits so CPU and GPU follow the same branches.
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetLogSpecialResultFromBits( int tBits, out bool tHasSpecialResult )
		{
			int tempAbsBits = tBits & 0x7FFFFFFF;
			int tempExp = ( tBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;
			int tempMant = tBits & MANTISSA_MASK;

			// Any NaN input returns NaN. The exact payload is intentionally not preserved.
			if ( tempExp == EXPONENT_ALL_ONES && tempMant != 0 )
			{
				tHasSpecialResult = true;
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			// log(+0) and log(-0) are both -infinity.
			if ( tempAbsBits == 0 )
			{
				tHasSpecialResult = true;
				return float.NegativeInfinity;
			}

			// log(negative finite) and log(-infinity) are NaN.
			if ( tBits < 0 )
			{
				tHasSpecialResult = true;
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			// log(+infinity) is +infinity.
			if ( tempAbsBits == 0x7F800000 )
			{
				tHasSpecialResult = true;
				return float.PositiveInfinity;
			}

			tHasSpecialResult = false;
			return 0.0f;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetSqrtSpecialResultFromBits( int tBits, out bool tHasSpecialResult )
		{
			int tempAbsBits = tBits & 0x7FFFFFFF;
			int tempExp = ( tBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;
			int tempMant = tBits & MANTISSA_MASK;

			// Any NaN input returns NaN. The exact payload is intentionally not preserved.
			if ( tempExp == EXPONENT_ALL_ONES && tempMant != 0 )
			{
				tHasSpecialResult = true;
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			// sqrt(+0) = +0 and sqrt(-0) = -0. Returning from bits preserves the sign.
			if ( tempAbsBits == 0 )
			{
				tHasSpecialResult = true;
				return BitConverter.Int32BitsToSingle( tBits );
			}

			// sqrt(negative finite) and sqrt(-infinity) are NaN.
			if ( tBits < 0 )
			{
				tHasSpecialResult = true;
				return BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			// sqrt(+infinity) is +infinity.
			if ( tempAbsBits == 0x7F800000 )
			{
				tHasSpecialResult = true;
				return float.PositiveInfinity;
			}

			tHasSpecialResult = false;
			return 0.0f;
		}

		// Extract the normalized mantissa and unbiased exponent used by log2.
		// Normal values are 1.mantissa * 2^exponent.
		// Subnormal values are mantissa * 2^-149 and must be normalized explicitly instead of clamped.
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static void GetNormalizedLogMantissaAndExponent( int tXBits, out int tMant, out int tExponent )
		{
			int tempExp = ( tXBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;
			int tempMant = tXBits & MANTISSA_MASK;

			if ( tempExp == 0 )
			{
				// Public log wrappers remove zero before this point. This guard keeps the helper safe.
				if ( tempMant == 0 )
				{
					tMant = 0;
					tExponent = int.MinValue;
					return;
				}

				// Locate the highest set bit in the 23-bit subnormal mantissa.
				int tempLeadingBit = MANTISSA_BITS - 1;

				while ( tempLeadingBit > 0 && ( ( tempMant & ( 1 << tempLeadingBit ) ) == 0 ) )
				{
					--tempLeadingBit;
				}

				// Shift the leading bit into the implicit-one position, then remove that implicit bit.
				int tempShiftLeft = MANTISSA_BITS - tempLeadingBit;
				tMant = ( tempMant << tempShiftLeft ) & MANTISSA_MASK;

				// After normalization: x = 1.tMant * 2^(-126 - shiftLeft).
				tExponent = -126 - tempShiftLeft;
				return;
			}

			tMant = tempMant;
			tExponent = tempExp - EXPONENT_BIAS;
		}

		// Log2(x) in Q32.32 via table + interpolation.
		// This function assumes x is positive, finite, and non-zero. Public wrappers enforce that.
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static long GetLog2DetQ3232FromBits( int tXBits, long[] tTableQ3232, int tIndexBits )
		{
			GetNormalizedLogMantissaAndExponent( tXBits, out int tempMant, out int tempE );

			int tempShift = MANTISSA_BITS - tIndexBits;
			int tempIndex = tempMant >> tempShift;
			int tempRem = tempMant & ( ( 1 << tempShift ) - 1 );
			long tempA = tTableQ3232[ tempIndex ];
			long tempB = tTableQ3232[ tempIndex + 1 ];
			long tempInterp = tempA + GetMulDivRoundToEvenS64( tempB - tempA, (uint)tempRem, tempShift );

			return ( (long)tempE << Q3232_FRACTION_BITS ) + tempInterp;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static long GetLog2DetQ3232FromBits( int tXBits, ArrayView<long> tTableQ3232, int tIndexBits )
		{
			GetNormalizedLogMantissaAndExponent( tXBits, out int tempMant, out int tempE );

			int tempShift = MANTISSA_BITS - tIndexBits;
			int tempIndex = tempMant >> tempShift;
			int tempRem = tempMant & ( ( 1 << tempShift ) - 1 );
			long tempA = tTableQ3232[ tempIndex ];
			long tempB = tTableQ3232[ tempIndex + 1 ];
			long tempInterp = tempA + GetMulDivRoundToEvenS64( tempB - tempA, (uint)tempRem, tempShift );

			return ( (long)tempE << Q3232_FRACTION_BITS ) + tempInterp;
		}

		// Exp2(z) from Q32.32 via table + interpolation => float
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetExp2DetFromSignedQ3232( long tZQ3232, uint[] tTableQ131, int tIndexBits )
		{
			bool tempIsNegative = tZQ3232 < 0;

			if ( tempIsNegative )
			{
				tZQ3232 = -tZQ3232;
			}

			float tempExp2 = GetExp2DetFromQ3232( tZQ3232, tTableQ131, tIndexBits );

			if ( tempIsNegative )
			{
				tempExp2 = GetReciprocalDet( tempExp2 );
			}

			return tempExp2;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetExp2DetFromSignedQ3232( long tZQ3232, ArrayView<uint> tTableQ131, int tIndexBits )
		{
			bool tempIsNegative = tZQ3232 < 0;

			if ( tempIsNegative )
			{
				tZQ3232 = -tZQ3232;
			}

			float tempExp2 = GetExp2DetFromQ3232( tZQ3232, tTableQ131, tIndexBits );

			if ( tempIsNegative )
			{
				tempExp2 = GetReciprocalDet( tempExp2 );
			}

			return tempExp2;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetExp2DetFromQ3232( long tZQ3232, uint[] tTableQ131, int tIndexBits )
		{
			int tempN = (int)( tZQ3232 >> Q3232_FRACTION_BITS );
			uint tempFrac = (uint)tZQ3232;
			int tempIndex = (int)( tempFrac >> ( Q3232_FRACTION_BITS - tIndexBits ) );
			uint tempRem = tempFrac & ( ( 1u << ( Q3232_FRACTION_BITS - tIndexBits ) ) - 1u );
			uint tempA = tTableQ131[ tempIndex ];
			uint tempB = ( tempIndex + 1 < tTableQ131.Length ) ? tTableQ131[ tempIndex + 1 ] : tempA;
			uint tempVQ131 = tempA + GetMulDivRoundToEvenU32( tempB - tempA, tempRem, Q3232_FRACTION_BITS - tIndexBits );
			uint tempMantQ = tempVQ131 - Q131_ONE;
			uint tempMant23 = GetShiftRightRoundToEvenU32( tempMantQ, Q131_TO_MANT23_SHIFT );
			int tempOutExp = tempN + EXPONENT_BIAS;

			if ( tempOutExp <= 0 )
			{
				return 0.0f;
			}
			else if ( tempOutExp >= ( 1 << EXPONENT_BITS ) - 1 )
			{
				return float.PositiveInfinity;
			}

			int tempOutBits = ( tempOutExp << EXPONENT_SHIFT ) | (int)( tempMant23 & MANTISSA_MASK );
			return BitConverter.Int32BitsToSingle( tempOutBits );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetExp2DetFromQ3232( long tZQ3232, ArrayView<uint> tTableQ131, int tIndexBits )
		{
			int tempN = (int)( tZQ3232 >> Q3232_FRACTION_BITS );
			uint tempFrac = (uint)tZQ3232;
			int tempIndex = (int)( tempFrac >> ( Q3232_FRACTION_BITS - tIndexBits ) );
			uint tempRem = tempFrac & ( ( 1u << ( Q3232_FRACTION_BITS - tIndexBits ) ) - 1u );
			uint tempA = tTableQ131[ tempIndex ];
			uint tempB = ( tempIndex + 1 < tTableQ131.Length ) ? tTableQ131[ tempIndex + 1 ] : tempA;
			uint tempVQ131 = tempA + GetMulDivRoundToEvenU32( tempB - tempA, tempRem, Q3232_FRACTION_BITS - tIndexBits );
			uint tempMantQ = tempVQ131 - Q131_ONE;
			uint tempMant23 = GetShiftRightRoundToEvenU32( tempMantQ, Q131_TO_MANT23_SHIFT );
			int tempOutExp = tempN + EXPONENT_BIAS;

			if ( tempOutExp <= 0 )
			{
				return 0.0f;
			}

			if ( tempOutExp >= ( 1 << EXPONENT_BITS ) - 1 )
			{
				return float.PositiveInfinity;
			}

			int tempOutBits = ( tempOutExp << EXPONENT_SHIFT ) | (int)( tempMant23 & MANTISSA_MASK );
			return BitConverter.Int32BitsToSingle( tempOutBits );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float GetReciprocalDet( float tX )
		{
			// Assume finite, tX > 0
			int tempBits = BitConverter.SingleToInt32Bits( tX );
			int tempExp = ( tempBits >> EXPONENT_SHIFT ) & EXPONENT_MASK;
			int tempMant = tempBits & MANTISSA_MASK;

			if ( tempExp == 0 )
			{
				return float.PositiveInfinity; // subnormal treated deterministically as +inf reciprocal
			}

			if ( tempExp == EXPONENT_ALL_ONES )
			{
				return ( tempMant == 0 ) ? 0.0f : BitConverter.Int32BitsToSingle( CANONICAL_NAN_BITS );
			}

			uint tempM = (uint)( tempMant | IMPLICIT_ONE_BIT );

			// Reciprocal mantissa in [1,2): compute q = round_to_even( (2^47) / tempM )
			// q is up to 2^23 * 2^24 = 2^47 scale, result q will be in [2^23, 2^24)
			ulong tempNum = 1ul << ( 2 * MANTISSA_BITS + 1 ); // 1<<47
			ulong tempQ = tempNum / tempM;
			ulong tempR = tempNum - ( tempQ * tempM );

			// Round-to-even on division remainder: compare 2*R with divisor
			ulong temp2R = tempR << 1;

			if ( temp2R > tempM || ( temp2R == tempM && ( tempQ & 1ul ) != 0ul ) )
			{
				++tempQ;
			}

			if ( tempQ >= ( 1ul << ( MANTISSA_BITS + 1 ) ) )
			{
				tempQ >>= 1;
				++tempExp;
			}

			int tempOutExp = ( ( 2 * EXPONENT_BIAS ) - 1 ) - tempExp;

			if ( tempOutExp <= 0 )
			{
				return 0.0f;
			}
			else if ( tempOutExp >= EXPONENT_ALL_ONES )
			{
				return float.PositiveInfinity;
			}

			uint tempOutMant = (uint)( tempQ & ( ( 1ul << MANTISSA_BITS ) - 1ul ) );
			int tempOutBits = ( tempOutExp << EXPONENT_SHIFT ) | (int)tempOutMant;

			return BitConverter.Int32BitsToSingle( tempOutBits );
		}

		// Q32.32 multiply with round-to-even (128-bit limb)
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static long GetMulQ3232( long tA, long tB )
		{
			bool tempIsNegative = ( tA ^ tB ) < 0;
			ulong tempUA = (ulong)( tA < 0 ? ( tA == long.MinValue ? long.MaxValue : -tA ) : tA );
			ulong tempUB = (ulong)( tB < 0 ? ( tB == long.MinValue ? long.MaxValue : -tB ) : tB );
			ulong tempA0 = tempUA & 0xFFFFFFFFul;
			ulong tempA1 = tempUA >> Q3232_FRACTION_BITS;
			ulong tempB0 = tempUB & 0xFFFFFFFFul;
			ulong tempB1 = tempUB >> Q3232_FRACTION_BITS;
			ulong tempP00 = tempA0 * tempB0;
			ulong tempP01 = tempA0 * tempB1;
			ulong tempP10 = tempA1 * tempB0;
			ulong tempP11 = tempA1 * tempB1;
			ulong tempMid = tempP01 + tempP10;
			ulong tempMidCarry = ( tempMid < tempP01 ) ? 1ul : 0ul;
			ulong tempLo = tempP00 + ( tempMid << Q3232_FRACTION_BITS );
			ulong tempLoCarry = ( tempLo < tempP00 ) ? 1ul : 0ul;
			ulong tempHi = tempP11 + ( tempMid >> Q3232_FRACTION_BITS ) + ( tempMidCarry << Q3232_FRACTION_BITS ) + tempLoCarry;
			ulong tempKept = ( tempHi << Q3232_FRACTION_BITS ) | ( tempLo >> Q3232_FRACTION_BITS );
			uint tempDiscarded = (uint)( tempLo & 0xFFFFFFFFul );
			ulong tempRounded = GetRoundToEvenKeepU64( tempKept, tempDiscarded );
			long tempResult = (long)tempRounded;

			if ( tempIsNegative )
			{
				tempResult = -tempResult;
			}

			return tempResult;
		}

		// Interpolation helpers (round-to-even)
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static long GetMulDivRoundToEvenS64( long tDelta, uint tRem, int tShift )
		{
			if ( tDelta == 0 || tRem == 0 )
			{
				return 0;
			}

			bool tempIsNegative = tDelta < 0;
			ulong tempDelta = (ulong)( tempIsNegative ? -tDelta : tDelta );
			ulong tempRemU = tRem;
			ulong tempD0 = tempDelta & 0xFFFFFFFFul;
			ulong tempD1 = tempDelta >> Q3232_FRACTION_BITS;
			ulong tempP0 = tempD0 * tempRemU;
			ulong tempP1 = tempD1 * tempRemU;
			ulong tempKept = ( tempP1 << ( 64 - tShift ) ) | ( tempP0 >> tShift );
			ulong tempDiscarded = tempP0 & ( ( 1ul << tShift ) - 1ul );
			ulong tempHalf = 1ul << ( tShift - 1 );

			if ( tempDiscarded > tempHalf || ( tempDiscarded == tempHalf && ( tempKept & 1ul ) != 0ul ) )
			{
				++tempKept;
			}

			long tempOut = (long)tempKept;

			if ( tempIsNegative )
			{
				tempOut = -tempOut;
			}

			return tempOut;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static uint GetMulDivRoundToEvenU32( uint tDelta, uint tRem, int tShift )
		{
			if ( tDelta == 0 || tRem == 0 )
			{
				return 0;
			}

			ulong tempProd = (ulong)tDelta * tRem;
			ulong tempKept = tempProd >> tShift;
			ulong tempDiscarded = tempProd & ( ( 1ul << tShift ) - 1ul );
			ulong tempHalf = 1ul << ( tShift - 1 );

			if ( tempDiscarded > tempHalf || ( tempDiscarded == tempHalf && ( tempKept & 1ul ) != 0ul ) )
			{
				++tempKept;
			}

			return (uint)tempKept;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static ulong GetRoundToEvenKeepU64( ulong tKept, uint tDiscarded32 )
		{
			if ( tDiscarded32 > DISCARDED_32_HALF )
			{
				return tKept + 1ul;
			}
			else if ( tDiscarded32 < DISCARDED_32_HALF )
			{
				return tKept;
			}

			return ( ( tKept & 1ul ) != 0ul ) ? ( tKept + 1ul ) : tKept;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static uint GetShiftRightRoundToEvenU32( uint tValue, int tShift )
		{
			if ( tShift <= 0 )
			{
				return tValue;
			}
			else if ( tShift >= 32 )
			{
				return 0;
			}

			uint tempShifted = tValue >> tShift;
			uint tempRem = tValue & ( ( 1u << tShift ) - 1u );
			uint tempHalf = 1u << ( tShift - 1 );

			if ( tempRem > tempHalf )
			{
				return tempShifted + 1u;
			}
			else if ( tempRem < tempHalf )
			{
				return tempShifted;
			}

			return ( ( tempShifted & 1u ) != 0u ) ? ( tempShifted + 1u ) : tempShifted;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static long GetShiftRightRoundToEvenS64( long tValue, int tShift )
		{
			if ( tShift <= 0 )
			{
				return tValue;
			}
			else if ( tShift >= 64 )
			{
				return 0;
			}

			long tempShifted = tValue >> tShift;
			long tempRem = tValue & ( ( 1L << tShift ) - 1L );
			long tempHalf = 1L << ( tShift - 1 );

			if ( tempRem > tempHalf )
			{
				return tempShifted + 1;
			}
			else if ( tempRem < tempHalf )
			{
				return tempShifted;
			}

			return ( ( tempShifted & 1L ) != 0L ) ? ( tempShifted + 1 ) : tempShifted;
		}
	}
}
