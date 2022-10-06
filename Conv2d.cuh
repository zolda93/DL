#ifndef _CONV2D_H
#define _CONV2D_H

#include "Layer.cuh"

namespace DL
{
	using namespace std;

	class Conv2d : public Layer
	{
		private:

			int number_filters;
			int filter_size;
			int stride;
			int padding ;
			int output_dim;

		public:
			Conv2d(int nf,int fz,int s=1,int p=0):number_filters(nf),filter_size(fz),stride(s),padding(p){}
			~Conv2d(){}


			virtual Tensor* forward(Tensor* x) override;
			virtual Tensor* backward(Tensor* dz) override;

			virtual void init_forward(Tensor* x) override;
			virtual void init_backward(Tensor* dz) override;
			virtual void init_params(unsigned int seed=0)override;
	};

	void Conv2d::init_forward(Tensor* x)
	{
		X = x;
		if(W == nullptr && b == nullptr)
		{
			W = new Tensor(number_filters,X->channels(),filter_size,filter_size);
			b = new Tensor(number_filters);
			init_params();
		}

		if(Z == nullptr)
		{
			output_dim = round((X->height() - filter_size + 2*padding)/stride + 1);
			Z = new Tensor(X->batch(),number_filters,output_dim,output_dim);
		}else if(Z->batch() != X->batch())
		{
			Z->reshape(X->batch(),Z->channels(),Z->height(),Z->width());
		}
	}

	void Conv2d::init_backward(Tensor* dz)
	{
		if( dW == nullptr && db == nullptr)
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


	void Conv2d::init_params(unsigned int seed)
	{
		random_device rd;
		mt19937 gen(seed == 0 ? rd() : static_cast<float>(seed));
		float rg = sqrt(6.f/((W->batch() + W->channels())*filter_size*filter_size));
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

	Tensor* Conv2d::forward(Tensor* x)
	{
		return convolution(Z,X,W,b,output_dim,number_filters,filter_size,stride);
	}

	Tensor* Conv2d::backward(Tensor* dz)
	{
		db_convolution(db,dz,number_filters);
		dW_convolution(dW,X,dz,filter_size,stride);
		if(stop_gradient)
		{
			return new Tensor();
		}else
		{
			return dX_convolution(dX,dz,W,stride);
		}
	}

}
#endif





















