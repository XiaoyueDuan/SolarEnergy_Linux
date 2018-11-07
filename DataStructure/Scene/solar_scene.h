//#pragma once
//
//#include "common_var.h"
//#include "grid.h"
//#include "heliostat.cuh"
//#include "random_generator.h"
//#include "receiver.cuh"
//#include "sunray.h"
//#include "destroy.h"
//#include "scene_instance_process.h"
//
////Singleton design model to  control the  access to resources
//class SolarScene {
//protected:
//	SolarScene();
//
//public:
//	static SolarScene* GetInstance();   //static member
//	static void InitInstance();
//	~SolarScene();
//
//	bool InitSolarScece();
//	bool InitSolarScene(string filepath);
//	bool LoadSceneFromFile(string filepath);
//
//	bool InitContent();					// Call the method only if all grids, heliostats and receivers needs initializing.
//
//private:
//	static SolarScene *m_instance;		//Singleton
//
//public:
//	float ground_length_;
//	float ground_width_;
//	int grid_num_;
//
//	SunRay *sunray_;
//	//scene object
//	vector<Grid *> grid0s;
//	vector<Heliostat *> heliostats;
//	vector<Receiver *> receivers;
//};
