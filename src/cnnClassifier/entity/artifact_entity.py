from dataclasses import dataclass

@dataclass
class ModelEvaluationArtifact:
    """
    Model evaluation stage ka output artifact.
    Isme trained model ka path aur evaluation results store hote hain.
    """
    trained_model_path: str
    test_accuracy: float = None
    is_model_accepted: bool = True


@dataclass
class ModelPusherArtifact:
    """
    Model pusher stage ka output artifact.
    Isme S3 bucket ka naam aur model ka S3 key path store hota hai.
    """
    bucket_name: str
    s3_model_path: str