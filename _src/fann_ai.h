#ifndef _FANN_AI_H
#define _FANN_AI_H

struct PlayerTrainingData;

class CFannAI : public CPlayerAI
{
	public:
		CFannAI() {}
		~CFannAI() {}
		void Init();
		void Think(COutputControl * playerKeys);
		void train_on(PlayerTrainingData *);

	protected:
		bool fromweight(float w);

		struct fann * nn;
};

#endif // _FANN_AI_H