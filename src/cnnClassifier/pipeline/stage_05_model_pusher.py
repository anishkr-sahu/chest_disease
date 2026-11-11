from cnnClassifier.components.model_pusher import ModelPusher
from cnnClassifier.entity.config_entity import ModelPusherConfig
from cnnClassifier.entity.artifact_entity import ModelEvaluationArtifact
from cnnClassifier import logging

STAGE_NAME = "Model Pusher stage"


class ModelPusherTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Evaluation artifact jisme trained model ka path ho
            model_eval_artifact = ModelEvaluationArtifact(
                trained_model_path="artifacts/training/model.h5",
                test_accuracy=0.95,
                is_model_accepted=True
            )
            
            model_pusher_config = ModelPusherConfig(
                bucket_name="chest-disease-bucket",
                s3_model_key_path="model.h5"
            )
            
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_eval_artifact,
                model_pusher_config=model_pusher_config
            )
            
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelPusherTrainingPipeline()
        pipeline.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx----------------x")
    except Exception as e:
        logging.exception(e)
        raise e