## Deep Tensor Generative Adversarial Nets


### Abstract
The promising deep generative models have been applied to many applications. 
  However, existing works reported generations of small images (e.g. 32 * 32 or 96 * 96). 
  In this paper, we propose a novel hierarchical deep tensor adversarial generative net (TGAN) that generates large high-quality images, e.g. 375 * 375. 
  First, we adopt tensor representation for images while existing works involve a vectorization preprocess which distorts the proximity information among pixels. Tensor representation properly captures the spatial patterns of elementary objects in images, leading to much concise features.
  Secondly, the compositional structure of images calls for hierarchical training schemes, therefore we propose a three-level adversarial process to generate images in a cascading manner. Essentially, the adversarial process takes place in a latent three-level tensor subspace.
  Thirdly, we show that in each level, the encoder of the generator learns the input distribution.  
  On the CIFAR10 dataset, the proposed TGAN achieves 4.05 inception score, outperforming state-of-the-art schemes.


### Dataset
* CIFAR10
* PASCAL2 VOC

### How to run it?
* Make sure the dataset you want to use is correct. If you want to use CIFAR10 dataset, modify the 29th line as :
	
		parser.add_argument('--dataset', default='CIFAR')

* if you want to use PASCAL2 VOC dataset, modify the 29th line as :

		parser.add_argument('--dataset', default='PASCAL')
* Then run train.py
* After finishing the training of the first layer,(suppose 70000 epoches), open train_second_layer.py, modify the 46th line as:

		netG.load_state_dict(torch.load('./1stLayer/1stLayerG69999.model'))
		
* After finishing the training of the first layer,(suppose 70000 epoches), open train_second_layer.py, modify the 50th line as:

		netG.load_state_dict(torch.load('./1stLayer/1stLayerG69999.model'))
        SecondG.load_state_dict(torch.load('./2ndLayer/2ndLayerG69999.model'))
        SecondE.load_state_dict(torch.load('./2ndLayer/2ndLayerE69999.model'))
        ThridE.load_state_dict(torch.load('./3rdLayer/3rdLayerE69999.model'))
        ThridG.load_state_dict(torch.load('./3rdLayer/3rdLayerG69999.model'))
        
