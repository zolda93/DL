#ifndef _POOLING_H
#define _POOLING_H

#include "Layer.cuh"
#include "cuTensor.cuh"

namespace DL
{
        class Pooling : public Layer
        {
                public:
                        Pooling(){}
                        ~Pooling(){}

                        virtual Tensor* forward(Tensor* x) override;
                        virtual Tensor* backward(Tensor* d) override;

                        virtual void init_forward(Tensor* x) override;
                        virtual void init_backward(Tensor* d) override;

        };

        void Pooling::init_forward(Tensor* x)
        {
                X = x;
                if(Z == nullptr)
                {
                        Z = new Tensor(X->batch(),X->channels(),X->height()/2,X->width()/2);
                }else if(Z->batch() != X->batch())
                {
                        Z->reshape(X->batch(),Z->channels(),Z->height(),Z->width());
                }
        }

        void Pooling::init_backward(Tensor* dz)
        {
                if(dX == nullptr)
                {
                        dX = new Tensor(X->shape());
                }else if(dX->batch() != X->batch())
                {
                        dX->reshape(X->batch());
                }
        }

        Tensor* Pooling::forward(Tensor* x)
        {
                return pooling_forward(Z,X);
        }

        Tensor* Pooling::backward(Tensor* dz)
        {
                return pooling_backward(dX,dz);
        }
}
#endif
