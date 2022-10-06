#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include <vector>
#include <utility>
#include "cuTensor.cuh"


namespace DL
{
	using namespace std;

	class Optimizer
	{
		protected:

			vector<pair<Tensor*,Tensor*>> weights;
			vector<pair<Tensor*,Tensor*>> biases;
		public:

			Optimizer(vector<Layer*> layers)
			{
				for(auto l:layers)
				{

					weights.emplace_back(make_pair(l->dW,l->W));
					biases.emplace_back(make_pair(l->db,l->b));
				}
			}


			virtual ~Optimizer(){}

			virtual void make_step()=0;
	};

	class SGD : public Optimizer
	{
		public:
			float  learning_rate;
			bool first;

			SGD(vector<Layer*> l,float lr):Optimizer(l),learning_rate(lr){}
			~SGD(){}

			virtual void make_step() override
			{
				for(int i=0; i < weights.size(); i++)
				{
					if(get<1>(weights[i]) == nullptr && get<1>(biases[i]) == nullptr)
						continue;
					SGD_step(get<0>(weights[i]),get<0>(biases[i]),get<1>(weights[i]),get<1>(biases[i]),learning_rate);
				}
				if(first == true)
					first = false;

			}
	};

	class Momentum : public Optimizer
	{
		public:
			Tensor* Vw = nullptr;
			Tensor* Vb = nullptr;
			float beta;
			float gamma;
			bool first = true;

			Momentum(vector<Layer*> l,float b,float g):Optimizer(l),beta(b),gamma(g){}

			~Momentum()
			{
				if (Vw != nullptr){ delete Vw; Vw = nullptr;}
				if(Vb != nullptr){delete Vb;Vb = nullptr;}
			}

			virtual void make_step() override
			{
				// for layer that does not have any weights ,just return
				for(int i=0;i<weights.size();i++)
				{
					if(get<1>(weights[i]) == nullptr && get<1>(biases[i]) == nullptr)
						continue;
					Momentum_step(get<0>(weights[i]),get<0>(biases[i]),Vw,Vb,get<1>(weights[i]),get<1>(biases[i]),beta,gamma);
				}
				
				if(first == true)
					first = false;
			}
	};


	class RMSProp : public Optimizer
	{
		public:
			Tensor* Sw = nullptr;
			Tensor* Sb = nullptr;
			float beta;
			float gamma;
			float epsilon;
			bool first = true;

			RMSProp(vector<Layer* > l,float b,float g,float e):Optimizer(l),beta(b),gamma(g),epsilon(e){}
			~RMSProp()
			{
				if(Sw != nullptr){delete Sw;Sw = nullptr;}
				if(Sb != nullptr){delete Sb;Sb = nullptr;}
			}

			virtual void make_step() override
			{

				for(int i=0;i<weights.size();i++)
				{
					if(get<1>(weights[i]) == nullptr && get<1>(biases[i]) == nullptr)
						continue;
					RMSProp_step(get<0>(weights[i]),get<0>(biases[i]),Sw,Sb,get<1>(weights[i]),get<1>(biases[i]),beta,gamma,epsilon);
				}

				if(first == true)
					first = false;

				//RMSProp_step(l->dW,l->db,Sw,Sb,l->W,l->b,beta,gamma,epsilon);
			}
	};


	class Adam : public Optimizer
	{
		public:

			Tensor* Vw = nullptr;
			Tensor* Vb = nullptr;
			Tensor* Sw = nullptr;
			Tensor* Sb = nullptr;
			float beta1;
			float beta2;
			float gamma;
			float epsilon;
			int iter = 0;

			Adam(vector<Layer*> l,float b1,float b2,float g,float e):Optimizer(l),beta1(b1),beta2(b2),gamma(g),epsilon(e){}
			~Adam()
			{
				if(Vw != nullptr){delete Vw,Vw = nullptr;}
				if(Vb != nullptr){delete Vb;Vb = nullptr;}
				if(Sw != nullptr){delete Sw;Sw = nullptr;}
				if(Sb != nullptr){delete Sb;Sb = nullptr;}
			}

			virtual void make_step() override
			{
				
				for(int i=0;i<weights.size();i++)
				{
					if(get<1>(weights[i]) == nullptr && get<1>(biases[i]) == nullptr)
						continue;
					if(iter == 0)
					{
						init_param_momentum(Vw,Vb,get<1>(weights[i]),get<1>(biases[i]));
						init_param_rmsprop(Sw,Sb,get<1>(weights[i]),get<1>(biases[i]));
					}

					Adam_step(get<0>(weights[i]),get<0>(biases[i]),Vw,Vb,Sw,Sb,get<1>(weights[i]),get<1>(biases[i]),beta1,beta2,gamma,epsilon,iter);
				}
				//Adam_step(l->dW,l->db,Vw,Vb,Sw,Sb,l->W,l->b,beta1,beta2,gamma,epsilon,iter);
				iter++;
			}
	};




}
#endif









