
__kernel void Hyst_filter(__global uchar4* inputImage, __global uchar4* outputImage)
{
	uint x = get_global_id(0);
	uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	float lowThresh = 20;
	float highThresh = 70;
	//float4 Gy = Gx;

	int c = x + y * width;


	const uchar EDGE = 255;

	uchar4 magnitude = inputImage[c];

	if (magnitude.x >= highThresh)
		outputImage[c] = EDGE;//convert_uchar4(EDGE);
	else if (magnitude.x <= lowThresh)
		outputImage[c] = 0;//convert_uchar4(0);
	else
	{
		float med = (highThresh + lowThresh) / 2;

		if (magnitude.x >= med)
			outputImage[c] = EDGE;//convert_uchar4(EDGE);
		else
			outputImage[c] = 0;//convert_uchar4(0);
	}

}