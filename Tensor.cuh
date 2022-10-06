#ifndef _TENSOR_H
#define _TENSOR_H


#include <array>
#include <iostream>
#include <assert.h>

#include "common.cuh"

namespace DL
{
	using namespace std;

	class Tensor
	{
		private:

			array<int,4> dim;
			float* device = nullptr;

		public:

			Tensor(int b=1,int c=1,int h=1,int w=1)
			{
				dim[0]=b;dim[1]=c;dim[2]=h;dim[3]=w;
				CHECK(cudaMalloc((void**)&device,lenght()*sizeof(float)));
			}

			Tensor(const array<int,4>& arr)
			{
				dim[0]=arr[0];dim[1]=arr[1];dim[2]=arr[2];dim[3]=arr[3];
				CHECK(cudaMalloc((void**)&device,lenght()*sizeof(float)));
			}

			~Tensor()
			{
				if(device != nullptr)
				{
					CHECK(cudaFree(device));
					device = nullptr;
				}
			}

			void reshape(int b=1,int c=1,int h=1,int w=1)
			{
				dim[0]=b;dim[1]=c;dim[2]=h;dim[3]=w;

				//assert(device != nullptr);

				CHECK(cudaFree(device));
				CHECK(cudaMalloc((void**)&device,lenght()*sizeof(float)));
			}

			void reshape(const array<int,4>& arr)
			{
				reshape(arr[0],arr[1],arr[2],arr[3]);
			}

			float* Device() const {return device;}
			int batch() const {return dim[0];}
			int channels() const{return dim[1];}
			int height() const{return dim[2];}
			int width() const {return dim[3];}
			int size() const{return dim[1]*dim[2]*dim[3];}
			int lenght() const{return dim[0]*dim[1]*dim[2]*dim[3];}	

			void set_shape(int b=1,int c=1,int h=1,int w=1)
			{
				dim[0]=b;dim[1]=c;dim[2]=h;dim[3]=w;
			}

			void set_shape(const array<int,4>& arr)
			{
				set_shape(arr[0],arr[1],arr[2],arr[3]);
			}

			array<int,4> shape() const {return dim;}

			bool same_shape(const array<int,4>& arr)
			{
				return dim==arr;
			}
			
			// copy data from cuda device to host
			void D2H(float* host,size_t size)
			{
				CHECK(cudaMemcpy(host,device,size*sizeof(float),cudaMemcpyDeviceToHost));
			}
			
			// copy data from host to cuda device
			void H2D(float* host,size_t size)
			{
				CHECK(cudaMemcpy(device,host,size*sizeof(float),cudaMemcpyHostToDevice));
			}

			void print_shape()
			{
				cout<<"Batch ="<<dec<<dim[0]<<" "<<"channels = "<<dec<<dim[1]<<" "<<"height = "<<dec<<dim[2]<<" "<<"width = "<<dec<<dim[3]<<endl;
			}

	};
}


#endif

