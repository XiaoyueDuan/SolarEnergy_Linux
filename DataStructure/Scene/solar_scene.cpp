
#include "solar_scene.h"
#include "scene_file_proc.h"

SolarScene* SolarScene::m_instance;
SolarScene* SolarScene::GetInstance()
{
	if (m_instance == NULL) InitInstance();
	return m_instance;
}

void SolarScene::InitInstance()
{
	m_instance = new SolarScene();
}

SolarScene::SolarScene() {
	//init the random
	RandomGenerator::initSeed();
	//init the sunray
	sunray_ = new SunRay(solarenergy::sun_dir,solarenergy::num_sunshape_groups,solarenergy::num_sunshape_lights_per_group,
		solarenergy::dni,solarenergy::csr);
	InitSolarScece();
}

SolarScene::~SolarScene() {
	// 1. free memory on GPU
	free_scene::gpu_free(receivers);
	free_scene::gpu_free(grid0s);
	free_scene::gpu_free(sunray_);

	// 2. free memory on CPU
	free_scene::cpu_free(receivers);
	free_scene::cpu_free(grid0s);
	free_scene::cpu_free(heliostats);
	free_scene::cpu_free(sunray_);
}

bool SolarScene::InitSolarScece() {
	string filepath = solarenergy::scene_filepath;
	return LoadSceneFromFile(filepath);
}

bool SolarScene::InitSolarScene(string filepath) {
	return LoadSceneFromFile(filepath);
}

bool SolarScene::LoadSceneFromFile(string filepath) {

	SceneFileProc proc;
	return proc.SceneFileRead(this, filepath);
}

bool SolarScene::InitContent()
{
	// 1. Sunray
	SceneProcessor::set_sunray_content(*this->sunray_);

	// 2. Grid
	SceneProcessor::set_grid_content(this->grid0s, this->heliostats);

	// 3. Receiver
	SceneProcessor::set_receiver_content(this->receivers);

	// 4. Heliostats
	float3 focus_center = this->receivers[0]->focus_center_;			// must after receiver init
	SceneProcessor::set_helio_content(this->heliostats, focus_center, this->sunray_->sun_dir_);

	return true;
}