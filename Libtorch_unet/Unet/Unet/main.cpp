#include"Project_information.h"

#define input_shape 512
#define num_classes 21

COLOR Classes[21]=
{
	{0,0,0},//background
	{128, 0, 0},//aeroplane
	{0, 128, 0},//bicycle
	{128, 128, 0},//bird
	{0, 0, 128},//boat
	{128, 0, 128},//bottle
	{0, 128, 128},//bus
	{128, 128, 128},//car
	{64, 0, 0},//cat
	{192, 0, 0},//chair
	{64, 128, 0},//cow
	{192, 128, 0},//diningtable
	{64, 0, 128},//dog
	{192, 0, 128},//horse
	{64, 128, 128},//motorbike
	{192, 128, 128},//person
	{0, 64, 0},//pottedplant
	{128, 64, 0},//sheep
	{0, 192, 0},//sofa
	{128, 192, 0},//train
	{0, 64, 128},//tvmonitor

};



int main(int argc, char** argv)
{
	Pro_info();
	if (argc < 1) return -1;
	std::string image_path = argv[1];
	//std::string model_path = argv[2];
	cv::Mat img;
	img = cv::imread(image_path);
	

	if (img.empty())
	{
		std::cerr << "The image can't open!\n";
		return -1;
	}
	

	//加载模型
	torch::jit::script::Module module;
	try
	{
		module = torch::jit::load("unet.pt");
	}
	catch (const c10::Error& e)
	{
		std::cerr << "The model can't load\n";
		return -1;
	}
	printf("The model load success!\n");

	//图像预处理
	cv::Mat input;
	cv::resize(img, img, cv::Size(input_shape, input_shape));//图片resize成512*512*3
	cv::cvtColor(img, input, cv::COLOR_BGR2RGB);
	

	//from_blob Mat转Tensor {batchsize,w,h,channles}
	torch::Tensor tensor_image = torch::from_blob(input.data, { 1,input.rows, input.cols,3 }, torch::kByte);
	
	//shape->(batchsize,channles,w,h)
	tensor_image = tensor_image.permute({ 0,3,1,2 });
	tensor_image = tensor_image.toType(torch::kFloat);

	//image/255.0图像的归一化处理
	tensor_image = tensor_image.div(255);

	//设置GPU,并将图像和模型放入GPU
	torch::DeviceType *deviceType = new torch::DeviceType();
	if (torch::cuda::is_available())
	{
		*deviceType = torch::kCUDA;
		std::cout << "The cuda is available" << std::endl;
	}
	else
	{
		*deviceType = torch::kCPU;
		std::cout << "The cuda isn't available" << std::endl;
	}
	
	torch::Device device(*deviceType);
	std::cout << *deviceType << std::endl;


	try
	{
		module.to(device);
		std::cout << "put model into the " << device << std::endl;
	}
	catch (const c10::Error& e)
	{
		std::cerr << "The model into gpu faill\n";
		return -1;
	}
	module.eval();

	torch::Tensor output;
	try
	{
		//如果不写tensor_image.to(device)在cuda环境下会报错，cpu推理可以正常进行
		output = module.forward({tensor_image.to(device)}).toTensor(); //The shape is [batch_size, num_classes, 512,512]
	}
	catch (const c10::Error& e)
	{
		std::cerr << "Can't get output!\n";
		return -1;
	}
	
	std::cout << "The output shape is: "<< output.sizes() << std::endl;
	auto tmp = output[0]; //[21,512,512]
	//std::cout << "The tmp shape is: " << tmp.sizes() << std::endl;
	auto pred = torch::softmax(tmp.permute({1,2,0}).detach(), -1).cpu().argmax(-1); //[512,512]

	torch::Tensor seg_img = torch::zeros({ input_shape,input_shape,3 });//准备接受语义分割图
	
	for (int c = 0; c < num_classes; c++)
	{

		seg_img.index({ "...", 0 }) += ((pred.index({ "..." }) == c) * Classes[c].B);
		seg_img.index({ "...", 1 }) += ((pred.index({ "..."}) == c) * Classes[c].G);
		seg_img.index({ "...", 2 }) += ((pred.index({ "..." }) == c) * Classes[c].R);

	}
	
	//在放入CPU的时候，必须要转uint8型，否则后面无法将tensor拷贝至Mat
	seg_img = seg_img.to(torch::kCPU).to(torch::kUInt8); 
	std::cout << "seq_img shape is " << seg_img.sizes() << std::endl;
	cv::Mat res(cv::Size(input_shape, input_shape), CV_8UC3,seg_img.data_ptr());
	
	cv::cvtColor(res,res,cv::COLOR_RGB2BGR);
	//std::cout << res << std::endl;
	cv::imshow("seg_img",res);
	
	if (cv::waitKey(0)&0xff==27)
	{
		cv::destroyAllWindows();
	}
	
	return 0;
}