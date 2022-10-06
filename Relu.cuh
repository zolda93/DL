#ifndef _RELU_H
#define _RELU_H

#include "Layer.cuh"

namespace DL
{
	class Relu : public Layer
	{
		public:
			Relu(){}
			~Relu(){}

			virtual Tensor* forward(Tensor* x) override;
			virtual Tensor* backward(Tensor* dz) override;

			virtual void init_forward(Tensor* x) override;
			virtual void init_backward(Tensor* dz) override;

	};

	void Relu::init_forward(Tensor* x)
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

	void Relu::init_backward(Tensor* dz)
	{
		if(dX == nullptr)
		{
			dX = new Tensor(X->shape());
		}else if(dX->batch() != X->batch())
		{
			dX->reshape(X->shape());
		}
	}

	Tensor* Relu::forward(Tensor* x)
	{
		return relu_forward(Z,X);
	}

	Tensor* Relu::backward(Tensor* dz)
	{
		return relu_backward(dX,dz,Z);
	}

}

#endif
