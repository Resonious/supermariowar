#ifdef _WIN32
#pragma comment(lib, "fannfloat.lib")
#endif

#include "fann.h"

#include "global.h"
#include "ai.h"
#include "fann_ai.h"
#include "dirlist.h"

void CFannAI::Init()
{
	CPlayerAI::Init();
	is_nn = true;

	nn = fann_create_standard(
	    3,  // layers
        14, // input
        7,  // hidden
		6   // output
	);
	fann_set_activation_steepness_layer(nn, 0.8, 2);
	fann_set_training_algorithm(nn, FANN_TRAIN_RPROP);

	DirectoryListing * files = new DirectoryListing("training/", ".data");
	if (!files->GetSuccess())
	{
		printf("FANN ERROR No training folder????\n");
		exit(1);
	}
	std::string filename;
	int filecount = 0;
	std::vector<const char*> filenames;
	while (files->operator()(filename))
	{
		printf("FANN training on %s\n", files->fullName(filename).c_str());
		filenames.push_back(files->fullName(filename).c_str());
		filecount += 1;
	}
	delete files;
	for (int i = 0; i < filecount; i++)
	{
		fann_train_on_file(
			nn, filenames[i],
			400, 50, 0.15
		);
	}

	if (filecount == 0)
		printf("FANN Small baby AI. No training files!.\n");
	else
		printf("FANN Trained on %i files.\n", filecount);
}

void CFannAI::train_on(PlayerTrainingData * data)
{
	float output[6] = {
		pPlayer->playerKeys->game_left.fDown ? 1.0 : 0.0,
		pPlayer->playerKeys->game_right.fDown ? 1.0 : 0.0,
		pPlayer->playerKeys->game_jump.fDown ? 1.0 : 0.0,
		pPlayer->playerKeys->game_down.fDown ? 1.0 : 0.0,
		pPlayer->playerKeys->game_turbo.fDown ? 1.0 : 0.0,
		pPlayer->playerKeys->game_powerup.fDown ? 1.0 : 0.0
	};

	// Train how many times?
	for (int i = 0; i < 300; i++)
		fann_train(nn, (float*)data, output);

	printf("FANN Learning...\n");
}

bool CFannAI::fromweight(float w)
{
	return w > 0.7;
}

void CFannAI::Think(COutputControl * playerKeys)
{
	playerKeys->game_left.fDown = false;
	playerKeys->game_right.fDown = false;
	playerKeys->game_jump.fDown = false;
	playerKeys->game_down.fDown = false;
	playerKeys->game_turbo.fDown = false;
	playerKeys->game_powerup.fDown = false;

	if(pPlayer->isdead() || pPlayer->isspawning())
		return;

	float * keys = fann_run(nn, (float*)&pPlayer->training);
	playerKeys->game_left.fDown    = fromweight(keys[0]);
	playerKeys->game_right.fDown   = fromweight(keys[1]);
	playerKeys->game_jump.fDown    = fromweight(keys[2]);
	playerKeys->game_down.fDown    = fromweight(keys[3]);
	playerKeys->game_turbo.fDown   = fromweight(keys[4]);
	playerKeys->game_powerup.fDown = fromweight(keys[5]);
}
