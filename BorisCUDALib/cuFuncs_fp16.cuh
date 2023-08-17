#pragma once

//conversion functions between float and double to/from half
//adapted from cuda_fp16.hpp

__device__ inline uint16_t float2half_as_uint16(const float a)
{
	uint16_t val;
	uint16_t r;

	unsigned int sign = 0U;
	unsigned int remainder = 0U;
	unsigned int x;
	unsigned int u;
	unsigned int result;

	memcpy(&x, &a, sizeof(a));

	u = (x & 0x7fffffffU);
	sign = ((x >> 16U) & 0x8000U);
	// NaN/+Inf/-Inf
	if (u >= 0x7f800000U) {
		remainder = 0U;
		result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
	}
	else if (u > 0x477fefffU) { // Overflows
		remainder = 0x80000000U;
		result = (sign | 0x7bffU);
	}
	else if (u >= 0x38800000U) { // Normal numbers
		remainder = u << 19U;
		u -= 0x38000000U;
		result = (sign | (u >> 13U));
	}
	else if (u < 0x33000001U) { // +0/-0
		remainder = u;
		result = sign;
	}
	else { // Denormal numbers
		const unsigned int exponent = u >> 23U;
		const unsigned int shift = 0x7eU - exponent;
		unsigned int mantissa = (u & 0x7fffffU);
		mantissa |= 0x800000U;
		remainder = mantissa << (32U - shift);
		result = (sign | (mantissa >> shift));
		result &= 0x0000FFFFU;
	}
	r = static_cast<unsigned short>(result);

	if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r & 0x1U) != 0U))) {
		r++;
	}
	val = r;

	return val;
}

__device__ inline float half_as_uint16_2float(const uint16_t h)
{
	float val;

	unsigned int sign = ((static_cast<unsigned int>(h) >> 15U) & 1U);
	unsigned int exponent = ((static_cast<unsigned int>(h) >> 10U) & 0x1fU);
	unsigned int mantissa = ((static_cast<unsigned int>(h) & 0x3ffU) << 13U);

	if (exponent == 0x1fU) { //NaN or Inf
		//discard sign of a NaN
		sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
		mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
		exponent = 0xffU;
	}
	else if (exponent == 0U) { //Denorm or Zero
		if (mantissa != 0U) {
			unsigned int msb;
			exponent = 0x71U;
			do {
				msb = (mantissa & 0x400000U);
				mantissa <<= 1U; //normalize
				--exponent;
			} while (msb == 0U);
			mantissa &= 0x7fffffU; //1.mantissa is implicit
		}
	}
	else {
		exponent += 0x70U;
	}
	const unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);

	memcpy(&val, &u, sizeof(u));

	return val;
}
