#ifndef _SOFTMAX_H
#define _SOFTMAX_H

#include "Layer.cuh"

namespace DL
{
	class Softmax : public Layer
	{
		public:

			Softmax(){}
			~Softmax(){}

			virtual Tensor* forward(Tensor* x) override;
			virtual Tensor* backward(Tensor* dz) override;

			virtual void init_forward(Tensor* x) override;
			virtual void init_backward(Tensor* dz) override;

	};

	void Softmax::init_forward(Tensor* x)
	{
		X = x;
		if(Z == nullptr)
		{
			Z = new Tensor(X->shape());
		}else if(Z->batch() != X->batch())
		{
			Z->reshape(X->batch());
		}
	}

	void Softmax::init_backward(Tensor* dz)
	{
		if(dX == nullptr)
		{
			dX = new Tensor(X->shape());
		}else if(dX->batch() != X->batch())
		{
			dX->reshape(X->shape());
		}
	}

	Tensor* Softmax::forward(Tensor* x)
	{
		return softmax_forward(Z,X);
	}

	Tensor* Softmax::backward(Tensor* dz)
	{
		//Tensor* Dz_wrt_Dx = Dz_wrt_x(Z);
		return softmax_backward(dX,dz,Z);
	}

}
#endif

