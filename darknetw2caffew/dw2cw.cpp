
#include "caffe/caffe.hpp"

using namespace caffe;

void loadweights(shared_ptr<Net<float> >& net, const char* argv);

int main(int argc, char** argv) {
	// Caffe (GPU)
	int device_id = 0;
	Caffe::SetDevice(device_id);
	Caffe::set_mode(Caffe::GPU);

	// Net
	boost::shared_ptr<Net<float> > net(new Net<float>("../models/yolo-face/yolo-face-deploy.prototxt", TEST));
	loadweights(net, "../models/yolo-face/yolo-face_final.weights");

	return 0;
}

void loadweights(boost::shared_ptr<Net<float> >& net, const char* argv){
	FILE *fp = fopen(argv, "rb");
	if(!fp)
		return;
	
	// read version
	int major;
	int minor;
	int revision;
	fread(&major, sizeof(int), 1, fp);
	fread(&minor, sizeof(int), 1, fp);
	fread(&revision, sizeof(int), 1, fp);
	if ((major*10 + minor) >= 2){
		size_t temp;
		fread(&temp, sizeof(size_t), 1, fp);
	} else {
		int iseen = 0;
		fread(&iseen, sizeof(int), 1, fp);
	}

	const std::vector<shared_ptr<Layer<float> > > layers = net->layers();
	int convolution_n = 0;
	int connect_n = 0;

	shared_ptr<Layer<float> > layer;
	std::vector<shared_ptr<Blob<float> > > blobs;

	for(int i = 0; i < layers.size(); ++i){
		layer = layers[i];
		blobs = layer->blobs();
		if(layer->type() == std::string("Convolution")) {
			++convolution_n;
			fread(blobs[1]->mutable_cpu_data(), sizeof(float), blobs[1]->count(), fp);
			fread(blobs[0]->mutable_cpu_data(), sizeof(float), blobs[0]->count(), fp);
		}
		else if(layer->type() == std::string("InnerProduct")) {
			++connect_n;
			fread(blobs[1]->mutable_cpu_data(), sizeof(float), blobs[1]->count(), fp);
			fread(blobs[0]->mutable_cpu_data(), sizeof(float), blobs[0]->count(), fp);
		}
	}
	
	fclose(fp);
}

