#ifndef _MODEL_H
#define _MODEL_H


#include <vector>
#include <string>
#include <fstream>
#include "Tensor.cuh"

namespace DL
{
	using namespace std;

	class Model
	{
		public:

			Model(){}
			~Model();

			void add_layer(Layer* l);
			void forward(Tensor* x);
			void backward(Tensor* target);

			int save_params(string filename);
			int load_params(string filename);
			void set_inference(){inference = true;}
			float loss(Tensor* target);

			vector<Layer*> model_layers() const{return layers;}
			Tensor* Output() const {return output;}
		private:

			vector<Layer*> layers;
			Tensor* output = nullptr;
			//string model_name;
			bool inference = false;
	};

	Model::~Model()
	{
		for(auto l:layers)
		{
			delete l;
		}
	}

	void Model::add_layer(Layer* l)
	{
		layers.push_back(l);
	}


	void Model::forward(Tensor* x)
	{
		Tensor* X = x;

		for(auto l:layers)
		{
			//if(!inference)
			//	l->init_forward(X);
			l->init_forward(X);
			X = l->forward(X);
		}

		layers.at(0)->set_stop_gradient();
		output = X;
		//return output;
	}

	void Model::backward(Tensor* target)
	{
		Tensor* grad = target;

		for(auto l = layers.rbegin(); l!=layers.rend();l++)
		{
			(*l)->init_backward(grad);
			grad = (*l)->backward(grad);
		}
	}

	int Model::save_params(string filename)
	{
		ofstream file(filename.c_str(),ios::out | ios::binary);
		if(!file.is_open())
		{
			cout<<"fail to open file"<<filename<<endl;
			return -1;
		}
		//int count = 0;

		for(auto l:layers)
		{
			if(l->W == nullptr && l->b == nullptr)
				continue;
			
			int weight_lenght = l->W->lenght();
			int bias_lenght = l->b->lenght();
			//int count = (weight_lenght + bias_lenght)*sizeof(float);
			//fssek(file,count,SEEK_SET);

			float* host_w = new float[weight_lenght*sizeof(float)];
			float* host_b = new float[bias_lenght*sizeof(float)];

			l->W->D2H(host_w,weight_lenght);
			l->b->D2H(host_b,bias_lenght);

			file.write((char*)host_w,weight_lenght*sizeof(float));

			//count += weight_lenght*sizeof(float);
			//fseek(file,count,SEEK_SET);

			file.write((char*)host_b,bias_lenght*sizeof(float));

			//count += bias_lenght*sizeof(float);

			delete[] host_w;
			delete[] host_b;
		}

		file.close();
		return 0;
	}

	int Model::load_params(string filename)
	{
		ifstream file(filename.c_str(),ios::in | ios::binary);

		if(!file.is_open())
		{
			cout<<"cant open file"<<endl;
			return -1;
		}

		//int count = 0;

		for(auto l:layers)
		{
			if(l->W == nullptr && l->b == nullptr)
				continue;
			
			char* host_w = new char[l->W->lenght()*sizeof(float)];
			char* host_b = new char[l->b->lenght()*sizeof(float)];

			//W = new Tensor(l->W->lenght());
			//b = new Tensor(l->b->lenght());

			//fseek(file,count,SEEK_SET);
			file.read((char*)host_w,l->W->lenght()*sizeof(float));
			//count += W->l->lenght()*sizeof(float);

			//fseek(file,count,SEEK_SET);
			file.read((char*)host_b,l->b->lenght()*sizeof(float));
			//count += l->b->lenght()*sizeof(float);

			l->W->H2D((float*)host_w,l->W->lenght());
			l->b->H2D((float*)host_b,l->b->lenght());

			delete[] host_w;
			delete[] host_b;
		}

		file.close();
		return 0;
	}

	float Model::loss(Tensor* target)
	{
		return layers.back()->loss(target);
	}



}

#endif







