import os
import sys

from cnnClassifier import logging
from cnnClassifier.cloud_storage.aws_storage import SimpleStorageService
from cnnClassifier.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from cnnClassifier.entity.config_entity import ModelPusherConfig
from cnnClassifier.entity.s3_estimator import ChestDiseaseEstimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        """
        :param model_evaluation_artifact: Output reference of model evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        try:
            self.s3 = SimpleStorageService()
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config

            # ChestDiseaseEstimator ka use karke S3 interaction handle karenge
            self.chest_estimator = ChestDiseaseEstimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_model_key_path,
            )
        except Exception as e:
            raise Exception(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function is used to upload the trained model to the S3 bucket
        
        Output      :   Returns ModelPusherArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            local_model_path = self.model_evaluation_artifact.trained_model_path
            
            # Check karein ki local model file exist karti hai ya nahi
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f"Trained model not found at path: {local_model_path}")

            logging.info(
                f"Uploading trained model from {local_model_path} to "
                f"s3://{self.model_pusher_config.bucket_name}/{self.model_pusher_config.s3_model_key_path}..."
            )

            # S3 par model upload karna (remove=False taaki local artifact safe rahe)
            self.chest_estimator.save_model(from_file=local_model_path, remove=False)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
            )

            logging.info("Successfully uploaded trained model to S3 bucket.")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise Exception(e, sys) from e