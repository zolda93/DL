#ifndef _LAYER_H
#define _LAYER_H

#include "cuTensor.cuh"

namespace DL
{
	class Layer
	{
		//friend class Model;
		//friend class Optimizer;
		protected:

			Tensor* X = nullptr;
			Tensor* Z = nullptr;

			Tensor* W = nullptr;
			Tensor* b = nullptr;

			Tensor* dX = nullptr;
			Tensor* dW = nullptr;
			Tensor* db = nullptr;

			bool stop_gradient = false;
			friend class Model;
			friend class Optimizer;

		public:

			Layer(){}
			virtual ~Layer();

			virtual Tensor* forward(Tensor* x)=0;
			virtual Tensor* backward(Tensor* dz)=0;

			virtual void init_forward(Tensor* x){}
			virtual void init_backward(Tensor* dz){}
			virtual void init_params(unsigned int seed=0){}
			virtual float loss(Tensor* target){return 0.f;}

			void set_stop_gradient(){stop_gradient = true;}
			//friend class Optimizer;
	};

	Layer::~Layer()
	{
		if(Z  != nullptr){delete Z  ; Z  = nullptr;}
		if(W  != nullptr){delete W  ; W  = nullptr;}
		if(b  != nullptr){delete b  ; b  = nullptr;}
		if(dX != nullptr){delete dX ; dX = nullptr;}
		if(dW != nullptr){delete dW ; dW = nullptr;}
		if(db != nullptr){delete db ; db = nullptr;}
	}
}
#endif

