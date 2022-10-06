#ifndef _SIGMOID_H
#define _SIGMOID_H

#include "Layer.cuh"

namesapace DL
{
	class Sigmoid : public Layer
	{
		public:
			Sigmoid(){}
			~Sigmoid(){}

			virtual Tensor* forward(Tensor* x) override;
			virtual Tensor* backward(Tensor* d) override;

			virtual void init_forward(Tensor* x) override;
			virtual void init_backward(Tensor* d) override;

	};

	void Sigmoid::init_forward(Tensor* x) 
	{
		X = x;
		if(Z == nullptr)
		{
			Z = new Tensor(X->shape());
		}else if(Z->batch() != X->batch())
		{
			Z->reshape(X->shape());
		}
	}

	void Sigmoid::init_backward(Tensor* dz)
	{
		if(dX == nullptr)
		{
			dX = new Tensor(X->shape());
		}else if(dX->batch() != X->batch())
		{
			dX->reshape(X->batch());
		}
	}

	Tensor* Sigmoid::forward(Tensor* x)
	{
		return sigmoid_forward(Z,X);
	}

	Tensor* Sigmoid::backward(Tensor* dz)
	{
		return sigmoid_backward(dX,dz,Z);
	}
}

#endif
