#ifndef _CUDAKERNEL_H
#define _CUDAKERNEL_H


#include <cmath>
#include <cstdlib>


namespace DL
{
	using namespace std;

	__global__ void dense_kernel(float* z,float* x,float* w,float* b,int batch,int width,int u)
	{
		unsigned int i = threadIdx.y + blockDim.y * blockIdx.x;
		unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int idx = i*u + j;

		float sum = 0.f;

		if(i < batch && j < u)
		{
			for(int k=0;k<width;k++)
			{
				sum += x[i*width + k]*w[k*u + j];
			}
			z[idx] = sum;
		}
	}

	__global__ void db_dense_kernel(float* db,float* dz,int batch,int size,int u)
	{
		unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

		float sum = 0.f;

		if(i < u)
		{
			for(int k=0;k<batch;k++)
			{
				sum += dz[k*size + i];
			}

			db[i] = sum/batch;
		}
	}

	__global__ void dW_dense_kernel(float* dw,float* x,float* dz,int batch,int size1,int size2,int u)
	{
		unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int idx =  i*u + j;

		float sum = 0.f;

		if(i<size1 && j<u)
		{
			for(int k=0;k<batch;k++)
			{
				sum += x[k*size1 + i]*dz[k*size2 + j];
			}

			dw[idx] = sum/batch;
		}
	}

	__global__ void dX_dense_kernel(float* dx,float* dz,float* w,size_t batch,int width,int s)
	{
		unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int idx = i*width + j;

		float sum = 0.f;

		if(i < batch && j < width)
		{
			for(int k=0;k<s;k++)
			{
				sum += dz[i*s + k]*w[j*s+k];
			}

			dx[idx] = sum/batch;
		}
	}


	__global__ void convolution_kernel(float* z,float* x,float* w,float* b,int c,int h,int kz,int s)
	{
		unsigned int i = threadIdx.y;
		unsigned int j = threadIdx.x;

		unsigned int _y = i + blockDim.y * blockIdx.y;
		unsigned int _x = j + blockDim.x * blockIdx.x;

		unsigned int idx = _y * gridDim.x * blockDim.x + _x;

		float sum = b[blockIdx.x];

		for(int k=0;k<c;k++)
		{
			for(int u=0;u<kz;u++)
			{
				for(int v=0;v<kz;v++)
				{
					if(i*s + u >= h || j*s + v >= h)
						continue;
					sum += x[blockIdx.y * h*h*c + k*h*h + (i*s + u)*h + j*s + v] * w[blockIdx.x*kz*kz*c + k*kz*kz + (kz-1 - u)*kz + kz-1-v] ;
				}
			}
		}

		z[idx] = sum;
		
	}

	__global__ void db_convolution_kernel(float* db,float* dz,int batch,int c,int h)
	{
		unsigned int idx = threadIdx.x;

		float sum = 0.f;

		for(int b=0;b<batch;b++)
		{
			for(int i=0;i<h;i++)
			{
				for(int j=0;j<h;j++)
				{
					sum += dz[b*c*h*h + idx*h*h + i*h+j];
				}
			}
		}

		db[idx] = sum/batch;
	}

	__global__ void dW_convolution_kernel(float* dw,float* x,float* dz,int batch,int XC,int XH,int dzC,int dzH,int s)
	{
		unsigned int i = threadIdx.y;
		unsigned int j = threadIdx.x;

		unsigned int _y = i + blockIdx.y * blockDim.y;
                unsigned int _x = j + blockIdx.x * blockDim.y;
                unsigned int idx = _y*blockDim.x*gridDim.x + _x;

		float sum =0.f;

		for(int b=0;b<batch;b++)
		{
			for(int u=0;u<dzH;u++)
			{
				for(int v=0;v<dzH;v++)
				{
					sum += x[b*XH*XH*XC + blockIdx.x * XH*XH + (XH-1-i-u*s)*XH + (XH-1-j-v*s)]*dz[b*dzH*dzH*dzC + blockIdx.y * dzH*dzH + (dzH-1-u)*dzH + (dzH-1-v)];
				}
			}
		}

		dw[idx] = sum/batch;
	}

	__global__ void dX_convolution_kernel(float* dx,float* dz,float* w,int xH,int C,int dzH,int wH,int s)
	{
		unsigned int i = threadIdx.y;
                unsigned int j = threadIdx.x;

		for(int q=0;q<C;q++)
                {
                        for(int u=0;u<wH;u++)
                        {
                                for(int v=0;v<wH;v++)
                                {

                                        dx[blockIdx.y*xH*xH*gridDim.x + blockIdx.x*xH*xH + (i*s + u)*xH + (j*s + v)] += \
                                        dz[blockIdx.y*dzH*dzH*C + q*dzH*dzH + i*dzH+j] * w[q*wH*wH*gridDim.x + blockIdx.x*wH*wH + (wH-1-u)*wH + (wH-1-v)];
                                }
                        }
                }
	}

	__global__ void transpose_convolution_kernel(float* z,float* x,float* w,int zh,int C,int xh,int wh,int s)
	{
		unsigned int i = threadIdx.y;
                unsigned int j = threadIdx.x;

                unsigned int i_o = i*s;
                unsigned int j_o = j*s;

		for(int c=0;c<C;c++)
		{
			for(int u=0;u<wh;u++)
			{
				for(int v=0;v<wh;v++)
				{
					if(j_o + v > zh-1 && i_o + v > zh-1)
						continue;
					z[blockIdx.y * zh*zh*gridDim.x + blockIdx.x*zh*zh + (i_o + u)*zh + j_o+v]+= x[blockIdx.y * xh*xh*C + c*xh*xh + i*xh + j] * w[c*wh*wh*gridDim.x + blockIdx.x*wh*wh + u*wh + v];
				}
			}
		}

	}


	__global__ void add_bias_kernel(float* z,float* b)
        {
                unsigned int i = threadIdx.y + blockIdx.y*blockDim.y;
                unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;

                unsigned int idx = i*gridDim.x*blockDim.x +j;

                z[idx] += b[blockIdx.x];
        }

	__global__ void db_transposed_convolution_kernel(float* db,float* dz,int batch,int c,int h)
	{
		unsigned int idx = threadIdx.x;
		float sum = 0.f;

		for(int b=0;b<batch;b++)
		{
			for(int i=0;i<h;i++)
			{
				for(int j=0;j<h;j++)
				{
					sum += dz[b*c*h*h + idx*h*h + i*h+j];
				}
			}
		}
		db[idx] = sum/batch;
	}


	__global__ void dW_transposed_convolution_kernel(float* dw,float* dz,float* x,int batch,int dzh,int zh,int s)
	{
		unsigned int i = threadIdx.y;
                unsigned int j = threadIdx.x;

                unsigned int _y = i + blockIdx.y * blockDim.y;
                unsigned int _x = j + blockIdx.x * blockDim.x;
                unsigned int idx = _y*gridDim.x*blockDim.x + _x;

		float sum = 0.f;

		for(int b=0;b<batch;b++)
                {
                        for(int u=0;u<zh;u++)
                        {

                                for(int v=0;v<zh;v++)
                                {
                                        if(j + v*s > dzh-1  || i + u*s > dzh-1)
						continue;
                                        
                                        sum+= dz[b*dzh*dzh*gridDim.x + blockIdx.x*dzh*dzh + (i+u*s)*dzh + j+v*s] * x[b*zh*zh*gridDim.y + blockIdx.y*zh*zh + u*zh +v] ;
                                }
                        }
                }

		dw[idx] = sum/batch;
	}


	__global__ void dX_transpose_convolution_kernel(float* dx,float* dz,float* w,int C,int dzh,int wh,int s)
	{
		unsigned int i = threadIdx.y;
                unsigned int j = threadIdx.x;

                unsigned int _y = i + blockIdx.y * blockDim.y;
                unsigned int _x = j + blockIdx.x * blockDim.x;

                unsigned int idx = _y*gridDim.x*blockDim.x + _x;

                float sum = 0.f;

		for(int c=0;c<C;c++)
                {
                        for(int u=0;u<wh;u++)
                        {
                                for(int v=0;v<wh;v++)
                                {
                                        if((i*s + u) > dzh-1 || (j*s + v) > dzh-1)
						continue;

					sum += dz[blockIdx.y*dzh*dzh*C + c*dzh*dzh + (i*s + u)*dzh + j*s + v] * w[blockIdx.x*wh*wh*C + c*wh*wh + u*wh+v];
                                }
                        }
                }

		dx[idx] = sum;
	}

	__global__ void relu_forward_kernel(float* z,float* x,int batch,int c,int h,int w)
	{
		if(h == 1 && w == 1)
                {

                        unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
                        unsigned int idx = i*c + j;

                        if(i < batch && j<c)
                        {
                                z[idx] = (x[idx] <= 0 ? 0 :  x[idx]);
                        }

                }else
                {
                        unsigned int i = threadIdx.y;
                        unsigned int j = threadIdx.x;

                        unsigned int _y = i + blockIdx.y * blockDim.y;
                        unsigned int _x = j + blockIdx.x*blockDim.x;

                        unsigned int idx = _y*gridDim.x*blockDim.x + _x;

                        z[idx] = (x[idx] <= 0 ? 0 :  x[idx]);
                }
	}


	__global__ void relu_backward_kernel(float* dx,float* dz,float* z,int batch,int c,int h,int w)
        {
                if(h == 1 && w == 1)
                {
                        unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
                        unsigned int idx = i*c + j;

                        if(i<batch && j<c)
                        {
                                dx[idx] = dz[idx] * ((z[idx] <= 0 ? 0 : 1));
                        }
                }else
                {
                        unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
                        unsigned int idx = i*gridDim.x*blockDim.x + j;

                        dx[idx] = dz[idx] * ((z[idx] <= 0 ? 0 : 1));
                }

        }

	__global__ void sigmoid_forward(float* z,float* x)
        {
                unsigned int i = threadIdx.y + blockIdx.y*blockDim.y;
                unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
                unsigned int idx = i*blockDim.x*gridDim.x + j;

                z[idx] = (1 / (1+exp(-x[idx])));
        }

       
	 __global__ void sigmoid_forward_kernel(float* z,float* x,int batch,int c,int h,int w)
        {
                if(h == 1 && w == 1)
                {
                        unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
                        unsigned int idx = i*c + j;

                        if(i<batch && j<c)
                        {
                                z[idx] = (1 / (1+exp(-x[idx])));
                        }
                }else
                {
                        unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
                        unsigned int idx = i*gridDim.x*blockDim.x + j;

                        z[idx] = (1 / (1+exp(-x[idx])));
                }

        }

	  __global__ void sigmoid_backward_kernel(float* dx,float* dz,float* z,int batch,int c,int h,int w)
        {
                if(h == 1 && w == 1)
                {
                        unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
                        unsigned int idx = i*c + j;

                        if(i<batch && j<c)
                        {
                                dx[idx] = dz[idx] * z[idx] * (1-z[idx]);
                        }
                }else
                {
                        unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
                        unsigned int idx = i*gridDim.x*blockDim.x + j;

                        dx[idx] = dz[idx] * z[idx] * (1-z[idx]);
                }

        }
	
	__global__ void tanh_forward_kernel(float* z,float* x,int batch,int c,int h,int w)
	{
		if( h == 1 && w == 1)
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;

			unsigned int idx = i*c + j;

			if(i < batch && j < c)
			{
				z[idx] = tanh(x[idx]);
			}
		}else
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*gridDim.x*blockDim.x + j;

			z[idx] = tanh(x[idx]);
		}
	}


	__global__ void tanh_backward_kernel(float* dx,float* dz,float* z,int batch,int c,int h,int w)
	{
		if(h == 1 && w == 1)
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*c + j;

			if(i < batch && j<c)
			{
				dx[idx] = dz[idx] * (1-z[idx]*z[idx]);
			}
		}else
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*gridDim.x * blockDim.x + j;

			dx[idx] = dz[idx] * (1-z[idx]*z[idx]);
		}
	}

	__global__ void pooling_forward_kernel(float* z,float* x)
	{
		unsigned int i = threadIdx.y + blockIdx.y*blockDim.y;
                 unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;

                 unsigned int idx = i*gridDim.x*blockDim.x + j;
                 float sum = 0.f;

                 for(int u=0;u<2;u++)
                 {
                         for(int v=0;v<2;v++)

                         {
                                 //unsigned int _idx = blockIdx.y * volume *4 + blockIdx.x * size*4   + (2i+u)*blockDim.x + (2j+v);
                                 sum +=  x[(2*i+u)*gridDim.x*blockDim.x*2 + 2*j+v];

                         }
                 }

                 z[idx] = sum/4;
	}

	__global__ void pooling_backward_kernel(float* dx,float* dz)
        {
                unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
                unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
                unsigned int idx = i*gridDim.x*blockDim.x + j;
                int d = ceil(i/2)*gridDim.x*blockDim.x/2 + ceil(j/2);

                dx[idx] = dz[d]/4;

        }

	__global__ void softmax_forward_kernel(float* z,float* x,int batch,int size)
	{
		unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int idx = i*size + j;

		float sum = 0.f;

		if(i < batch && j < size)
		{
			for(int k=0;k<size;k++)
			{
				sum += exp(x[i*size + k]);
			}

			z[idx] = exp(x[idx]) / sum;
		}
	}

	__global__ void softmax_backward_kernel(float* dx,float* z,float* target,int batch,int size)
	{
		unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int idx = i*size + j;

		if(i <batch && j<size)
		{

			dx[idx] = z[idx] - target[idx];
		}
	}

	__global__ void Zeros_kernel(float* x,int batch,int c,int h,int w)
	{
		if( h == 1 && w == 1)
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*c + j;

			if(i<batch && j<c)
			{
				x[idx] = 0.f;
			}
		}else
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*gridDim.x*blockDim.x + j;
			x[idx] = 0.f;
		}
	
	}

	__global__ void sgd_kernel_W(float* dW,float* W,float learning_rate,int batch,int c,int h,int w)
	{
		if(h == 1 && w == 1)
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*c + j;

			if(i < batch && j < c)
			{
				
				W[idx] -= learning_rate * dW[idx];
			}
		}else
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*gridDim.x*blockDim.x + j;
			W[idx] -= learning_rate * dW[idx];
		}
	}

	__global__ void momentum_kernel_W(float* dW,float* W,float* Vw,float beta,float gamma,int batch,int c,int h,int w)
	{
		if(h == 1 && w == 1)
		{
			unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*c + j;

			if(i < batch && j <c)
			{
				float prev_Vw = Vw[idx];

				__syncthreads();

				Vw[idx] = beta * prev_Vw + (1-beta)*dW[idx];

				__syncthreads();

				W[idx] -= gamma * Vw[idx];
			}
		}else
		{
			unsigned int i = threadIdx.y + blockDim.y + blockIdx.y;
			unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
			unsigned int idx = i*gridDim.x * blockDim.x + j;

			float prev_Vw = Vw[idx];

			__syncthreads();

			Vw[idx] = beta * prev_Vw + (1-beta)*dW[idx];

			__syncthreads();

			W[idx] -= gamma * Vw[idx];
		}
	}

	__global__ void rmsprop_kernel_W(float* dW,float* W,float* Sw,float beta,float gamma,float epsilon,int batch,int c,int h,int w)
	{
		if(h == 1 && w == 1)
                {
                        unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
                        unsigned int idx = i*c + j;

                        if(i < batch && j <c)
                        {
				float prev_Sw = Sw[idx];

				__syncthreads();

                                Sw[idx] = beta * prev_Sw + (1-beta) * dW[idx] * dW[idx];

				__syncthreads();

                                W[idx] -= gamma * dW[idx] * (1/sqrt(Sw[idx] + epsilon));
                        }
                }else
                {
                        unsigned int i = threadIdx.y + blockDim.y + blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
                        unsigned int idx = i*gridDim.x * blockDim.x + j;
			
			float prev_Sw = Sw[idx];

			__syncthreads();

                        Sw[idx] = beta * prev_Sw + (1-beta) * dW[idx] * dW[idx];

			__syncthreads();

                        W[idx] -= gamma * dW[idx] * (1/sqrt(Sw[idx] + epsilon));
                }
	}

	__global__ void adam_kernel_W(float* dW,float* W,float* Vw,float* Sw,float beta1,float beta2,float gamma,float epsilon,int iter,int batch,int c,int h,int w)
	{
		if(h == 1 && w == 1)
                {
                        unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
                        unsigned int idx = i*c + j;

                        if(i < batch && j <c)
                        {
				
				float prev_Vw = Vw[idx];
				float prev_Sw = Sw[idx];

				__syncthreads();

				Vw[idx] =(beta1 * prev_Vw + (1-beta1) * dW[idx]) * (1/(1-pow(beta1,iter)));
                                Sw[idx] =(beta2 * prev_Sw + (1-beta2) * dW[idx] * dW[idx]) *(1/ (1-pow(beta2,iter))); 

				__syncthreads();

                                W[idx] -= gamma * Vw[idx] * (1/sqrt(Sw[idx] + epsilon));
                        }
                }else
                {
                        unsigned int i = threadIdx.y + blockDim.y + blockIdx.y;
                        unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
                        unsigned int idx = i*gridDim.x * blockDim.x + j;

			
			float prev_Vw = Vw[idx];
                        float prev_Sw = Sw[idx];

                        __syncthreads();

                        Vw[idx] =(beta1 * prev_Vw + (1-beta1)*dW[idx]) * (1/(1-pow(beta1,iter)));
                        Sw[idx] =(beta2 * prev_Sw + (1-beta2) * dW[idx] * dW[idx]) *(1/(1-pow(beta2,iter)));

			__syncthreads();

                        W[idx] -= gamma * Vw[idx] * (1/sqrt(Sw[idx] + epsilon));
                }
	}

	__global__ void sgd_kernel_b(float* db,float* b,float learning_rate,int lenght)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

		if(idx < lenght)
		{
			b[idx] -= learning_rate * db[idx];
		}
	}

	__global__ void momentum_kernel_b(float* db,float* b,float* Vb,float beta,float gamma,int lenght)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

		if(idx <lenght)
		{
			float prev_Vb = Vb[idx];

			__syncthreads();

			Vb[idx] = beta * prev_Vb + (1-beta) * db[idx];

			__syncthreads();

			b[idx]  -= gamma * Vb[idx];
		}
	}

	__global__ void rmsprop_kernel_b(float* db,float* b,float* Sb,float beta,float gamma,float epsilon,int lenght)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

		if(idx <lenght)
		{
			float prev_Sb = Sb[idx];

			__syncthreads();

			Sb[idx] = beta * prev_Sb + (1-beta) * db[idx] * db[idx];

			__syncthreads();

			b[idx] -= gamma * db[idx] * (1/sqrt(Sb[idx] + epsilon));
		}
	}

	__global__ void adam_kernel_b(float* db,float* b,float* Vb,float* Sb,float beta1,float beta2,float gamma,float epsilon,int iter,int lenght)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	


		if(idx < lenght)
		{
			
			float prev_Vb = Vb[idx];
			float prev_Sb = Sb[idx];

			__syncthreads();

			Vb[idx] = (beta1 * prev_Vb + (1-beta1) * db[idx]) * (1/(1-pow(beta1,iter)));
			Sb[idx] = (beta2 * prev_Sb + (1-beta2) * db[idx] * db[idx]) * (1/(1-pow(beta2,iter)));

			__syncthreads();

			b[idx] -= gamma * Vb[idx] * (1/sqrt(Sb[idx] + epsilon));
		}
	}

	__global__ void crossentropy_kernel(float* z,float* target,float* dloss,int size)
	{
		unsigned int idx = threadIdx.x;

		float sum = 0.f;

		for(int k=0;k<size;k++)
		{
			sum -= target[idx*size + k]*log(z[idx*size + k]);
		}

		dloss[idx] = sum;
	}



}

#endif




