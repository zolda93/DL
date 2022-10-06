#ifndef _DENSE_H
#define _DENSE_H

#include <cmath>
#include <random>
#include "Layer.cuh"

namespace DL
{

	using namespace std;

	class Dense : public Layer
	{
		private:
			int units;
		public:
			Dense(int u):units(u){}

			~Dense(){}

			virtual Tensor* forward(Tensor* x) override;
			virtual Tensor* backward(Tensor* dz) override;

			virtual void init_forward(Tensor* x) override;
			virtual void init_backward(Tensor* dz) override;
			virtual void init_params(unsigned int seed=0) override;
	};

	void Dense::init_forward(Tensor* x)
	{
		X = x;

		if(W == nullptr && b == nullptr)
		{
			W = new Tensor(X->size(),units);
			b = new Tensor(units);
			init_params();
		}

		if(Z == nullptr)
		{
			Z = new Tensor(X->batch(),units);
		}else if(Z->batch() != X->batch())
		{
			Z->reshape(X->batch(),units);
		}

	}

	void Dense::init_backward(Tensor* dz)
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


	void Dense::init_params(unsigned int seed)
	{
		random_device rd;
		mt19937 gen(seed==0 ? rd():static_cast<unsigned int>(seed));
		float rg = sqrt(6.f/(W->batch() + units));
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


	Tensor* Dense::forward(Tensor* x)
	{
		return dense(Z,X,W,b,units);
	}

	Tensor* Dense::backward(Tensor* dz)
	{
		db_dense(db,dz,units);
		dW_dense(dW,X,dz,units);

		if(stop_gradient)
		{
			return new Tensor();
		}else{
			return dx_dense(dX,dz,W);
		}
	}

}

#endif



