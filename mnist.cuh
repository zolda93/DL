#ifndef _MNIST_H
#define _MNIST_H


#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <assert.h>

#include "Tensor.cuh"

namespace DL
{
	using namespace std;

	class MNIST
	{
		private:
			string dataset_dir;

			string train_dataset = "train-images-idx3-ubyte";
                        string train_labels  = "train-labels-idx1-ubyte";
                        string test_dataset  = "t10k-images-idx3-ubyte";
                        string test_labels   = "t10k-labels-idx1-ubyte";

			float* data_pool = nullptr;
			float* target_pool = nullptr;

			Tensor* data_batch = nullptr;
			Tensor* target_batch = nullptr;

			void load_data(string& image_file_path);
			void load_target(string& label_file_path);

			int to_int(uint8_t* ptr);

			int step = 0;
			int batch_size = 1;
			int channels = 1;
			int height = 1;
			int width = 1;
			int num_samples;

			int num_steps;
			int quotion ;
			

		public:

			MNIST():dataset_dir("./"){}
			MNIST(string dir):dataset_dir(dir){}
			~MNIST();

			void train(int batch_size=1);
			void test(int batch_size=1);

			void get_batch();

			int next();

			Tensor* get_data() const{return data_batch;}
			Tensor* get_label() const{return target_batch;}
	};

	MNIST::~MNIST()
	{
		cout<<"-----START FREE MNIST----"<<endl;
		if (data_batch != nullptr){delete data_batch;data_batch=nullptr;}
                if(target_batch != nullptr){delete target_batch;target_batch = nullptr;}
                if(data_pool != nullptr){delete[] data_pool;data_pool = nullptr;}
                if(target_pool != nullptr){delete[] target_pool;target_pool = nullptr;}

                cout<<"--------FREE MNIST-------"<<endl;
	}

	void MNIST::load_data(string& image_file_path)
	{
		uint8_t ptr[4];

		string file_path = dataset_dir + "/" + image_file_path;

                cout<<"loading..."<<file_path<<endl;

                ifstream file(file_path.c_str(),ios::binary|ios::in);

                if(!file.is_open())
                {
                        cout<<"Download dataset first"<<endl;
                        exit(-1);
                }

		file.read((char*)ptr,4);
                int magic = to_int(ptr);
                assert((magic & 0xFFF) == 0x803);

                file.read((char*)ptr,4);
                num_samples = to_int(ptr);
                file.read((char*)ptr,4);
                height = to_int(ptr);
                file.read((char*)ptr,4);
                width = to_int(ptr);

		data_pool =new float[num_samples*channels*height*width*sizeof(float)];

		uint8_t* q = new uint8_t[num_samples*height*width];
		file.read((char*)q,num_samples*width*height);

		for(int i=0;i<num_samples;i++)
		{
			for(int j=0;j<height*width;j++)
			{
				data_pool[i*height*width + j ] = (float)(q[i*height*width + j])/255.f;
			}
			
		}


		delete[] q;
	}

	void MNIST::load_target(string& label_file_path)
	{
		uint8_t ptr[4];

                string file_path = dataset_dir + "/" + label_file_path;

                ifstream file(file_path.c_str(),ios::binary|ios::in);

                if(!file.is_open())
                {
                        cout<<"ERROR CHECK DATASET LABEL"<<endl;
                        exit(-1);
                }

                file.read((char*)ptr,4);
                int magic = to_int(ptr);
                assert((magic & 0xFFF) == 0x801);

                file.read((char*)ptr,4);
                int num_target = to_int(ptr);

                target_pool = new float[num_target*10];

                for(int i=0;i<num_target;i++)
                {
                        float target_sample[10];

                        fill(target_sample,&target_sample[10],0.f);
			file.read((char*)ptr,1);
                        target_sample[static_cast<int>(ptr[0])]=1.f;
                        for(int j=0;j<10;j++)
                        {
                                target_pool[i*10 + j] = target_sample[j];
                        }

                }

		//for(int i=0;i<32;i++)
		//{
		//	for(int j=0;j<10;j++)
		//	{
		//		cout<<target_pool[i*10 + j]<<" ";
		//	}

		//	cout<<endl;
		//}

                file.close();
	}

	int MNIST::to_int(uint8_t* ptr)
        {

                return ((ptr[0] & 0xFF) << 24 | (ptr[1] & 0xFF) << 16 |
                        (ptr[2] & 0xFF) << 8 | (ptr[3] & 0xFF) << 0);
        }


	void MNIST::train(int batch)
        {
                batch_size = batch;

                load_data(train_dataset);
                load_target(train_labels);

		num_steps = num_samples / batch_size;
		quotion = num_samples % batch_size;

        }

        void MNIST::test(int batch)
        {
                batch_size = batch;

                load_data(test_dataset);
                load_target(test_labels);

                num_steps = num_samples / batch_size;
		quotion = num_samples % batch_size;

        }

	void MNIST::get_batch()
	{
		int data_idx;
		int target_idx;

		if(step < num_steps)
		{
			data_idx = step*batch_size*height*width;
			target_idx = step*batch_size*10;

			if(data_batch == nullptr)
			{
				data_batch = new Tensor(batch_size,channels,height,width);
			}
			if(target_batch == nullptr)
			{
				target_batch = new Tensor(batch_size,10);
			}

		}else if(step >= num_steps && step < num_steps + 1)
		{
			if(quotion > 0)
			{
				//cout<<"-------RESHAPE-------"<<endl;
				data_idx = num_steps*batch_size*height*width;
				target_idx = num_steps * batch_size*10;
				data_batch->reshape(quotion,channels,height,width);
				target_batch->reshape(quotion,10);
			}else
			{
				return;
			}
			
		}

		float* dhost   = &data_pool[data_idx];
		float* htarget = &target_pool[target_idx];
		//cout<<"-----------------------------START COPYING DATA TO DEVICE---------------------"<<endl;

		data_batch->H2D(dhost,data_batch->lenght());
                target_batch->H2D(htarget,target_batch->lenght());

		//cout<<"----------------------------END COPYING DATA TO DEVICE-------------------------"<<endl;

	}

	int MNIST::next()
	{
		step++;
		get_batch();
		return step;
	}

}

#endif

			




























