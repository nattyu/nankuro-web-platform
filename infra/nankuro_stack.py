from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as _lambda,
    aws_apigatewayv2 as apigw,
    aws_apigatewayv2_integrations as integrations,
    aws_ecr_assets as ecr_assets,
    CfnOutput
)
from constructs import Construct
import os

class NankuroStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Build and Push Docker Image to ECR (handled by CDK Assets)
        # Assumes the Dockerfile is at project root (or we point to it)
        # Our Dockerfile is in `lambda_app/Dockerfile` but context should be project root
        # to carry `solver_core` as well.
        # But `lambda_app/Dockerfile` expects `solver_core` relative to root.
        # So context = "..", but standard CDK usage is easier if we point to root.
        
        # Path to project root from here (infra/)
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        docker_image = _lambda.DockerImageCode.from_image_asset(
            directory=root_dir,
            file="lambda_app/Dockerfile"
        )

        # 2. Lambda Function
        solver_fn = _lambda.DockerImageFunction(
            self, "NankuroSolverFn",
            code=docker_image,
            memory_size=3008, # Reverted to 3GB due to AWS Quota limits
            timeout=Duration.seconds(120), 
            architecture=_lambda.Architecture.X86_64,
        )

        # Enable Function URL (Auth=NONE, CORS enabled)
        fn_url = solver_fn.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE,
            cors=_lambda.FunctionUrlCorsOptions(
                allowed_origins=["*"],
                allowed_methods=[_lambda.HttpMethod.ALL],
                allowed_headers=["*"],
            )
        )
        
        CfnOutput(self, "FunctionUrl", value=fn_url.url)

        # 3. HTTP API Gateway
        http_api = apigw.HttpApi(
            self, "NankuroApi",
            cors_preflight={
                "allow_origins": ["*"],
                "allow_methods": [apigw.CorsHttpMethod.POST],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        )

        # Integration
        integration = integrations.HttpLambdaIntegration(
            "SolverIntegration",
            solver_fn
        )

        # Routes
        http_api.add_routes(
            path="/api/solve",
            methods=[apigw.HttpMethod.POST],
            integration=integration
        )
        http_api.add_routes(
            path="/api/ocr",
            methods=[apigw.HttpMethod.POST],
            integration=integration
        )

        # Outputs
        CfnOutput(self, "ApiUrl", value=http_api.url)
        CfnOutput(self, "LambdaArn", value=solver_fn.function_arn)
