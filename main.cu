
#include "Dense.cuh"
#include "Relu.cuh"
#include "Model.cuh"
#include "mnist.cuh"
#include "Softmax.cuh"
#include "Conv2d.cuh"
#include "Pooling.cuh"
#include "Loss.cuh"
#include "Optimizer.cuh"

using namespace DL;
using namespace std;
int main()
{
	int batch_size = 32;
	MNIST mnist("./dataset");
	mnist.train(batch_size);
	Model model;
	model.add_layer(new Conv2d(6,3));
	model.add_layer(new Relu());
	model.add_layer(new Pooling());
	model.add_layer(new Conv2d(12,3));
	model.add_layer(new Relu());
	model.add_layer(new Pooling());
	model.add_layer(new Dense(10));
	model.add_layer(new Softmax());
	model.add_layer(new CrossEntropyLoss());

	Optimizer* sgd = new SGD(model.model_layers(),0.0001);
	
	int epochs = 0;
	
	while(epochs < 1)
	{
		int step = 0;
		mnist.get_batch();
		while(step < 234)
        	{
                	model.forward(mnist.get_data());
                	float loss = model.loss(mnist.get_label());
                	model.backward(mnist.get_label());
                	sgd->make_step();

                	if(step % 100 == 0)
                	{
                        	cout<<"----loss----="<<loss<<endl;
                        	
                	}
                	step = mnist.next();
		}
		epochs++;
	}

		

	return 0;
}


