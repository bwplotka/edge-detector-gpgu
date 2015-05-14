__constant float gaus[3][3] = { { 0.0625, 0.125, 0.0625 },
{ 0.1250, 0.250, 0.1250 },
{ 0.0625, 0.125, 0.0625 } };

__kernel void gaussian_filter(__global uchar4* inputImage, __global uchar4* outputImage)
{
	uint x = get_global_id(0);
	uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	float4 Gx = (float4)(0);
	//float4 Gy = Gx;

	int c = x + y * width;

	
	/* Read each texel component and calculate the filtered value using neighbouring texel components */
	if (x >= 1 && x < (width - 1) && y >= 1 && y < height - 1)
	{
		float4 i00 = convert_float4(inputImage[c - 1 - width]);
		float4 i10 = convert_float4(inputImage[c - width]);
		float4 i20 = convert_float4(inputImage[c + 1 - width]);
		float4 i01 = convert_float4(inputImage[c - 1]);
		float4 i11 = convert_float4(inputImage[c]);
		float4 i21 = convert_float4(inputImage[c + 1]);
		float4 i02 = convert_float4(inputImage[c - 1 + width]);
		float4 i12 = convert_float4(inputImage[c + width]);
		float4 i22 = convert_float4(inputImage[c + 1 + width]);

		Gx = i00*gaus[0][0] + i10*gaus[1][0] + i20*gaus[2][0] + i01*gaus[0][1] +i11*gaus[1][1] + i21*gaus[2][1] + i02*gaus[0][2] + i12*gaus[1][2] + i22*gaus[2][2];


		outputImage[c] = convert_uchar4(Gx);

	}



}
