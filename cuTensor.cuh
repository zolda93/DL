#ifndef _CUTENSOR_H
#define _CUTENSOR_H

#include "Tensor.cuh"
#include "cudaKernel.cuh"


using namespace DL;


#define cuda_grid_block(b,c,h,w)\
	{\
		if (h == 1 && w == 1){\
			dim3 block(c <= 32 ? c:32,b<=32 ? b:32);\
			dim3 grid((c + block.x - 1 )/block.x,(b + block.y - 1)/block.y);\
		}else\
		{\
			dim3 block(w,h);\
			dim3 grid(c,b);\
		}\
	}\

Tensor* dense(Tensor* Z,Tensor* X,Tensor* W,Tensor* b,int units)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(Z->batch(),Z->channels(),Z->height(),Z->width());
	dense_kernel<<<grid,block>>>(Z->Device(),X->Device(),W->Device(),b->Device(),X->batch(),X->size(),units);

	return Z;
}



void db_dense(Tensor* db,Tensor* dz,int units)
{
	dim3 block(units <= 1024 ? units:1024);
	db_dense_kernel<<<1,block>>>(db->Device(),dz->Device(),dz->batch(),dz->channels(),units);
}

void dW_dense(Tensor* dW,Tensor* X,Tensor* dz,int units)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dW->batch(),dW->channels(),dW->height(),dW->width());
	dW_dense_kernel<<<grid,block>>>(dW->Device(),X->Device(),dz->Device(),X->batch(),X->size(),dz->size(),dW->size());
}

Tensor* dx_dense(Tensor* dX,Tensor* dz,Tensor* W)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channels(),dX->height(),dX->width());
	dX_dense_kernel<<<grid,block>>>(dX->Device(),dz->Device(),W->Device(),dX->batch(),dX->size(),W->size());
	return dX;
}

Tensor* convolution(Tensor* Z,Tensor* X,Tensor* W,Tensor* b,int output_dim,int number_filters,int filter_size,int stride)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(X->batch(),numbers_filters,output_dim,output_dim);
	convolution_kernel<<<grid,block>>>(Z->Device(),X->Device(),W->Device(),b->Device(),X->channels(),X->height(),filter_size,stride);
	return Z;
}

void db_convolution(Tensor* db,Tensor* dz,int number_filters)
{
	db_convolution_kernel<<<1,number_filters>>>(db->Device(),dz->Device(),dz->batch(),dz->channels(),dz->height());
}

void dW_convolution(Tensor* dW,Tensor* X,Tensor* dz,int filter_size,int stride)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dW->batch(),dW->channels(),dW->height(),dW->width());
	dW_convolution_kernel<<<grid,block>>>(dW->Device(),X->Device(),dz->Device(),X->batch(),X->channels(),X->height(),dz->channels(),dz->height(),stride);
}

Tensor* dX_convolution(Tensor* dX,Tensor* dz,Tensor* W,int stride)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channels(),dX->height(),dX->width());
	dX_convolution_kernel<<<grid,block>>>(dX->Device(),dz->Device(),W->Device(),dX->height(),dz->channels(),dz->height(),W->height(),stride);
	return dX;
}

Tensor* transpose_convolution(Tensor* Z,Tensor* X,Tensor* W,Tensor* b,int stride,int number_filters)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(Z->batch(),Z->channels(),X->height(),X->width());
	transpose_convolution_kernel<<<grid,block>>>(Z->Device(),X->Device(),W->Device(),Z->height(),number_filters,X->height(),W->height(),stride);

	dim3 blockb(Z->width(),Z->height());
	dim3 gridb(Z->channels(),Z->batch());
	add_bias_kernel<<<gridb,blockb>>>(Z->Device(),b->Device());

	return Z;
}

void db_transpose_convolution(Tensor* db,Tensor* dz)
{
	db_transposed_convolution_kernel<<<1,dz->channels()>>>(db->Device(),dz->Device(),dz->batch(),dz->channels(),dz->height());
}

void dW_transposed_convolution(Tensor* dW,Tensor* dz,Tensor* X,int stride)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dW->batch(),dW->channel(),dW->height(),dW->width());
	dW_transposed_convolution_kernel<<<grid,block>>>(dW->Device(),dz->Device(),X->Device(),dz->batch(),dz->height(),X->height(),stride);
}

Tensor* dX_transpose_convolution(Tensor* dX,Tensor* dz,Tensor* W,int stride)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channels(),dX->height(),dX->width());
	dX_transpose_convolution_kernel<<<grid,block>>>(dX->Device(),dz->Device(),W->Device(),dz->channels(),dz->height(),W->height(),stride);
	return dX;

}

Tensor* relu_forward(Tensor* Z,Tensor* X)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(Z->batch(),Z->channels(),Z->height(),Z->width());
	relu_forward_kernel<<<grid,block>>>(Z->Device(),X->Device(),Z->batch(),Z->channels(),Z->height(),Z->width());

	return Z;
}

Tensor* relu_backward(Tensor* dX,Tensor* dz,Tensor* Z)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channels(),dX->height(),dX->width());
	relu_backward_kernel<<<grid,block>>>(dX->Device(),dz->Device(),Z->Device(),dX->batch(),dX->channels(),dX->height(),dX->width());

	return dX;

}

Tensor* sigmoid_forward(Tensor* Z,Tensor* X)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(Z->batch(),Z->channels(),Z->height(),Z->width());
	sigmoid_forward_kernel<<<grid,block>>>(Z->Device(),X->Device(),Z->batch(),Z->channels(),Z->height(),Z->width());

        return Z;

}


Tensor* sigmoid_backward(Tensor* dX,Tensor* dz,Tensor* Z)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channels(),dX->height(),dX->width());
	sigmoid_backward_kernel<<<grid,block>>>(dX->Device(),dz->Device(),Z->Device(),dX->batch(),dX->channels(),dX->height(),dX->width());

        return dX;
}

Tensor* tanh_forward(Tensor* Z,Tensor* X)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(Z->batch(),Z->channels(),Z->height(),Z->width());
	tanh_forward_kernel<<<grid,block>>>(Z->Device(),X->Device(),Z->batch(),Z->channels(),Z->height(),Z->width());

        return Z;
}


Tensor* tanh_backward(Tensor* dX,Tensor* dz,Tensor* Z)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channels(),dX->height(),dX->width());
	tanh_backward_kernel<<<grid,block>>>(dX->Device(),dz->Device(),Z->Device(),dX->batch(),dX->channels(),dX->height(),dX->width());

        return dX;
}

Tensor* pooling_forward(Tensor* Z,Tensor* X)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(Z->batch(),Z->channels(),Z->height(),Z->width());
	pooling_forward_kernel<<<grid,block>>>(Z->Device(),X->Device());

	return Z;
}

Tensor* pooling_backward(Tensor* dX,Tensor* dz)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channel(),dX->height(),dX->width());
	pooling_backward_kernel<<<grid,block>>>(dX->Device(),dz->Device());
	return dX;
}

Tensor* softmax_forward(Tensor* Z,Tensor* X)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(Z->batch(),Z->channels(),Z->height(),Z->width());
	softmax_forward_kernel<<<grid,block>>>(Z->Device(),X->Device(),Z->batch(),Z->size());
	return Z;
}

Tensor* softmax_backward(Tensor* dX,Tensor* Z,Tensor* target)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(dX->batch(),dX->channels(),dX->height(),dX->width());
	softmax_backward_kernel<<<grid,block>>>(dX->Device(),Z->Device(),target->Device(),dX->batch(),dX->size());
	return dX;
}

void Zeros(Tensor* X)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(X->batch(),X->channels(),X->height(),X->width());
	Zeros_kernel<<<grid,block>>>(X->Device(),X->batch(),X->channels(),X->height(),X->width());
}



void init_param_momentum(Tensor* Vw,Tensor* Vb,Tensor* W,Tensor* b)
{
	//CHECK(cudaMalloc((void**)&Vw->Device(),lenght_w*sizeof(float)));
	//CHECK(cudaMalloc((void**)&Vb->Device(),lenght_b*sizeof(float)));
	Vw = new Tensor(W->shape());
	Vb = new Tensor(b->shape());
	Zeros(Vw);
	Zeros(Vb);
}

void init_param_rmsprop(Tensor* Sw,Tensor* Sb,Tensor* W,Tensor* b)
{
	Sw = new Tensor(W->shape());
	Sb = new Tensor(b->shape());
	Zeros(Sw);
	Zeros(Sb);
}

void SGD_step(Tensor* dW,Tensor* db,Tensor* W,Tensor* b,float learning_rate)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(W->batch(),W->channels(),W->height(),W->width());
	sgd_kernel_W<<<grid,block>>>(dW->Device(),W->Device(),learning_rate,W->batch(),W->channels(),W->height(),W->width());

	dim3 blockb(b->lenght() <= 1024 ? b->lenght():1024);
	dim3 gridb((b->lenght() + blockb.x - 1)/blockb.x);
	sgd_kernel_b<<<gridb,blockb>>>(db->Device(),b->Device(),learning_rate,b->lenght());
}

void Momentum_step(Tensor* dW,Tensor* db,Tensor* Vw,Tensor* Vb,Tensor* W,Tensor* b,float beta,float gamma)
{
	dim3 block;
        dim3 grid;
        cuda_grid_block(W->batch(),W->channels(),W->height(),W->width());
        momentum_kernel_W<<<grid,block>>>(dW->Device(),W->Device(),Vw->Device(),beta,gamma,W->batch(),W->channels(),W->height(),W->width());

	dim3 blockb(b->lenght() <= 1024 ? b->lenght():1024);
	dim3 gridb((b->lenght() + blockb.x - 1)/blockb.x);
	momentum_kernel_b<<<gridb,blockb>>>(db->Device(),b->Device(),Vb->Device(),beta,gamma,b->lenght());
}

void RMSProp_step(Tensor* dW,Tensor* db,Tensor* Sw,Tensor* Sb,Tensor* W,Tensor* b,float beta,float gamma,float epsilon)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(W->batch(),W->channels(),W->height(),W->width());
	rmsprop_kernel_W<<<grid,block>>>(dW->Device(),W->Device(),Sw->Device(),beta,gamma,epsilon,W->batch(),W->channels(),W->height(),W->width());

	dim3 blockb(b->lenght() <= 1024 ? b->lenght():1024);
	dim3 gridb((b->lenght() + blockb.x - 1)/blockb.x);
	rmsprop_kernel_b<<<gridb,blockb>>>(db->Device(),b->Device(),Sb->Device(),beta,gamma,epsilon,b->lenght());
}

void Adam_step(Tensor* dW,Tensor* db,Tensor* Vw,Tensor* Vb,Tensor* Sw,Tensor* Sb,Tensor* W,Tensor* b,float beta1,float beta2,float gamma,float epsilon,int iter)
{
	dim3 block;
	dim3 grid;
	cuda_grid_block(W->batch(),W->channels(),W->height(),W->width());
	adam_kernel_W<<<grid,block>>>(dW->Device(),W->Device(),Vw->Device(),Sw->Device(),beta1,beta2,gamma,epsilon,iter,W->batch(),W->channels(),W->height(),W->width());

	dim3 blockb(b->lenght() <= 1024 ? b->lenght():1024);
	dim3 gridb((b->lenght() + blockb.x - 1)/blockb.x);
	adam_kernel_b<<<gridb,blockb>>>(db->Device(),b->Device(),Vb->Device(),Sb->Device(),beta1,beta2,gamma,epsilon,iter,b->lenght());
}

void crossentropyloss(Tensor* Z,Tensor* target,Tensor* dloss)
{
	crossentropy_kernel<<<1,Z->batch()>>>(Z->Device(),target->Device(),dloss->Device(),Z->size());
}




#endif
			




		




			































