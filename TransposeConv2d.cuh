#ifndef _TRANSPOSECONV_H
#define _TRANSPOSECONV_H

#include "Layer.cuh"


namespace DL
{
	using namespace std;

	class Transpconv2d : public Layer
	{
		private:

			int output_channels;
			int number_filters;
			int filter_size;
			int stride;
			int padding;
			int output_dim;
		public:
			Transpconv2d(int out,int channels,int nf,int fz,int s=1,int p=0)
			{
				output_dim = out;
				output_channels = channels;
				number_filters = nf;
				filter_size = fz;
				stride = s;
				padding = p;
			}

			~Transpconv2d(){}

			virtual Tensor* forward(Tensor* x) override;
			virtual Tensor* backward(Tensor* dz) override;

			virtual void init_forward(Tensor* x) override;
			virtual void init_backward(Tensor* dz) override;
			virtual void init_params(unsigned int sedd=0) override;
	};

	void Transpconv2d::init_forward(Tensor* x)
	{
		X = x;

		if(W == nullptr && b == nullptr)
		{
			W = new Tensor(X->channels(),output_channels,filter_size,filter_size);
			b = new Tensor(output_channels);
			init_params();
		}

		if(Z == nullptr)
		{
			Z = new Tensor(X->batch(),output_channels,output_dim,output_dim);
		}else if(Z->batch() != X->batch())
		{
			Z->reshape(X->batch(),Z->channels(),Z->height(),Z->width());
		}
	}

	void Transpconv2d::init_backward(Tensor* dz)
	{
		if(dW == nullptr && db == nullptr)
		{
			dW = new Tensor(W->shape());
			db = new Tensor(b->shape());
		}
		if(dX == nullptr)
		{
			dX = new Tensor(X->shape());
		}else if(dX->batch() != X->batch())
		{
			dX->reshape(X->shape());
		}
	}


	void Transpconv2d::init_params(unsigned int seed)
	{
		random_device rd;
		mt19937 gen(seed == 0 ? rd() : static_cast<float>(seed));
		float rg = sqrt(6.f / ((W->batch() + W->channels())*filter_size*filter_size));
		uniform_real_distribution<> dis(-rg,rg);

		float* _w = new float[W->lenght()];
		float* _b = new float[b->lenght()];


		for(int i=0;i<W->lenght();i++)
			_w[i] = static_cast<float>(dis(gen));

		for(int i=0;i<b->lenght();i++)
			_b[i] = 0.f;

		W->H2D(_w,W->lenght());
		b->H2D(_b,b->lenght());

		delete[] _w;
		delete[] _b;
	}


	Tensor* Transpconv2d::forward(Tensor* x)
	{
		return transpose_convolution(Z,X,W,b,stride,number_filters);
	}


	Tensor* Transpconv2d::backward(Tensor* dz)
	{
		db_transpose_convolution(db,dz);
		dW_transposed_convolution(dW,dz,X,stride);
		if(stop_gradient)
		{
			return new Tensor();
		}else{
			return dX_transpose_convolution(dX,dz,W,stride);
		}
	}

}

#endif

