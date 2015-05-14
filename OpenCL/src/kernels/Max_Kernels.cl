/**********************************************************************
Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/*
* For a description of the algorithm and the terms used, please see the
* documentation for this sample.
*
* Each thread calculates a pixel component(rgba), by applying a filter
* on group of 8 neighbouring pixels in both x and y directions.
* Both filters are summed (vector sum) to form the final result.
*/


__kernel void Max_filter(__global uchar4* inputImage, __global uchar4* outputImage, __global uchar* theta)
{
	uint x = get_global_id(0);
	uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);


	int c = x + y * width;

	uchar4 magnitude = inputImage[c];

	switch (theta[c])
	{
	case 0:
		if (magnitude.x <= inputImage[c - 1].x || magnitude.x <= inputImage[c + 1].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;

	case 45:
		if (magnitude.x <= inputImage[c + 1 - width].x || magnitude.x <= inputImage[c - 1 + width].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;

	case 90:
		if (magnitude.x <= inputImage[c - width].x || magnitude.x <= inputImage[c + width].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;

	case 135:
		if (magnitude.x <= inputImage[c - 1 - width].x || magnitude.x <= inputImage[c + 1 + width].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;
	}



}