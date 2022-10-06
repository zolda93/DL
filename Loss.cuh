#ifndef _LOSS_H
#define _LOSS_H


#include "cuTensor.cuh"
#include "Layer.cuh"


namespace DL
{
	using namespace std;

	class CrossEntropyLoss : public Layer
	{
		public:

			CrossEntropyLoss(){}

			~CrossEntropyLoss()
			{
				if(dloss != nullptr){delete dloss;dloss = nullptr;}
				if(hloss != nullptr){delete[]  hloss;hloss = nullptr;}
			}


			virtual Tensor* forward(Tensor* x) override;
			virtual Tensor* backward(Tensor* dz) override;

			virtual void init_forward(Tensor* x)override;
			virtual void init_backward(Tensor* dz)override;

			virtual float loss(Tensor* y) override;

			Tensor* dloss = nullptr;
			float* hloss = nullptr;
	};

	void CrossEntropyLoss::init_forward(Tensor* x)
	{
		X = x;
	}

	void CrossEntropyLoss::init_backward(Tensor* dz)
	{
		dX = dz;
	}

	Tensor* CrossEntropyLoss::forward(Tensor* x)
	{
		return X;
	}

	Tensor* CrossEntropyLoss::backward(Tensor* dz)
	{
		return dX;
	}

	float CrossEntropyLoss::loss(Tensor* y)
	{
		if (dloss == nullptr)
		{
			dloss = new Tensor(X->batch());
		}

		if(hloss == nullptr)
		{
			hloss = new float[X->batch()];
		}

		crossentropyloss(X,y,dloss);

		dloss->D2H(hloss,X->batch());
		float result = 0.f;

		for(int i=0;i<X->batch();i++)
		{
			result += hloss[i];
		}
		return result/(X->batch());

	}
}

#endif

		






