import json
import os.path

from src._types import SavedModelTypes
from src.analysis.pipelines.analysis_pipeline import AnalysisPipeline, AnalysisPipelineSteps
from src.analysis.pipelines.evaluation_pipeline import EvaluationPipeline, EvaluationPipelineSteps
from src.analysis.pipelines.pipeline import SavedModel
from src.analysis.pipelines.preperation_pipeline import PreparationPipeline, PreparationSteps
from variables import DATA_DIR, MS_TYPES_TO_MAIN_TYPE, MODEL_DIR


def run_preparation_pipeline(pipeline_config):
    preparationPipeline = PreparationPipeline(**pipeline_config)
    preparationPipeline.save_single_images = False
    preparationPipeline.use_hidden_initial_labels = True
    preparationPipeline.num_model_samples = [32, 32, 32, 32, 32] # [200, 200, 200, 200, 200] # [200, 29, 3, 160, 8] #[100, 15, 2, 79, 4] # [200, 29, 3, 160, 8]
    preparationPipeline.run()

def run_evaluation_pipeline(pipeline_config):
    pipeline_config["saved_clf"] = SavedModel(
        model_name=SavedModelTypes.clf,
        checkpoint_path= os.path.join(
            MODEL_DIR,
            "analysis_final_ms_clf_base_20250711_101044_best"
        ),
    )
    evaluationPipeline = EvaluationPipeline(**pipeline_config)
    evaluationPipeline.map_labels_prec_rec = MS_TYPES_TO_MAIN_TYPE
    evaluationPipeline.use_hidden_initial_labels = True
    evaluationPipeline.batch_size = 4
    evaluationPipeline.run()

def run_analysis_pipeline(pipeline_config):
    if "saved_clf" in pipeline_config.keys():
        del pipeline_config["saved_clf"]
    analysis_pipeline = AnalysisPipeline(**pipeline_config)
    analysis_pipeline.use_hidden_initial_labels = True
    # Use for CVAE as cluster_prototypes is not supported yet
    # analysis_pipeline.supported_steps = [AnalysisPipelineSteps.original_geometrics, AnalysisPipelineSteps.model_geometrics, AnalysisPipelineSteps.proto, AnalysisPipelineSteps.clusters]
    analysis_pipeline.run()

def run_full_evaluation_for_checkpoints(base_name: str, checkpoint_dir: str, json_config_path: str):
    """
    Run preparation and evaluation pipelines for all checkpoints saved in best_checkpoints.json
    :param checkpoint_dir:
    :param base_name:
    :return:
    """
    with open(json_config_path, 'r') as file:
        data = json.load(file)
    for checkpoint in data[base_name]:
        if isinstance(checkpoint, dict):
            config_kwargs = {k: checkpoint[k] for k in checkpoint.keys() - {'checkpoint'}}
            if config_kwargs.get("skip"):
                print("Skipping {}".format(checkpoint))
                continue
            else:
                print("Running {}".format(checkpoint))
            if config_kwargs.get("latent_name") is not None:
                config_kwargs["latent_infer_path"] = os.path.join(checkpoint_dir, config_kwargs["latent_name"])

            pipeline_config = {
                "saved_model": SavedModel(
                    model_name= SavedModelTypes(config_kwargs["name"]),
                    config_kwargs= config_kwargs if len(config_kwargs) > 0 else None,
                    checkpoint_path= os.path.join(
                        checkpoint_dir,
                        str(checkpoint["checkpoint"])
                    )
                ),
                "base_path": os.path.join(DATA_DIR, "analysis_data"),
                "device": "cuda"
            }
            run_preparation_pipeline(pipeline_config)
            run_evaluation_pipeline(pipeline_config)
            run_analysis_pipeline(pipeline_config)
        else:
            raise Exception(f"Unknown checkpoint config {checkpoint}")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    # HINT: Examples for creating analysis data for multiple models (add them to a json and let them be analysed automatically)
    # base_path = MODEL_DIR
    # run_full_evaluation_for_checkpoints("bdae", base_path, 'bdae_comparison_checkpoints_scaled+shift.json')

    # base_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "cond_encoder_scale")
    # run_full_evaluation_for_checkpoints("bdae", base_path, 'bdae_comparison_checkpoints_scaled.json')

    # base_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "bdae_class_norm")
    # run_full_evaluation_for_checkpoints("bdae", base_path, 'bdae_comparison_checkpoints_znormalize.json')

    # base_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "final_cvae")
    # run_full_evaluation_for_checkpoints("cvae", base_path, 'cvae_comparison_checkpoints.json')
    # pipeline_config = {
    #     "saved_model": SavedModel(
    #         model_name=SavedModelTypes(),
    #         # config_kwargs={"layers": 10, "hidden_ch": 1024, "wrs": False, "latent_infer_path": os.path.join(base_path, "latents_no_wrs.pkl")},
    #         config_kwargs={"layers": 10, "hidden_ch": 2048, "wrs": False, "num_classes": 5, "latent_infer_path": os.path.join(base_path, "latents_class_spec_norm.pkl")},
    #         # config_kwargs={"layers": 10, "hidden_ch": 2048, "wrs": False, "num_classes": 5, "num_classes_base_model": 5, "latent_infer_path": os.path.join(base_path, "latents_cond_encoder_class_spec_norm.pkl")},
    #         # config_kwargs={"kld": 10, "ch": 64, "wrs": False, "num_classes": 5},
    #         checkpoint_path= os.path.join(
    #             base_path,
    #             # "latent_wrs_layers10_hidden1024_lossLossType.l1_latentdiffusion_20250715_202621_best"
    #             # "latent_no_wrs_class_spec_norm_layers10_hidden2048_lossLossType.l1_latentdiffusion_20250719_112843_best"
    #             # "latent_no_wrs_class_spec_norm_layers_add_target_shift_10_hidden2048_lossLossType.l1_latentdiffusion_20250720_110901_best"
    #             # "final_cvae_5_classes_ch64_kld20_base_20250719_112945_best"
    #         )
    #     ),
    #     "base_path": os.path.join(DATA_DIR, "analysis_data"),
    #     "device": "cuda"
    # }

    pipeline_config = {
        "saved_model": SavedModel(
            model_name=SavedModelTypes.bdae_20_1024_5_cond_encoder_shift_scale_alpha,
            config_kwargs={"layers": 20, "hidden_ch": 1024, "num_classes": 5, "model_conf_num_classes": 5, "latent_net_conf_num_classes": 5, "class_znormalize": True, "enc_merge_time_and_cond_embedding": True, "scale_target_alpha": True, "shift_target": True, "latent_infer_path": os.path.join(MODEL_DIR, "final_latents_cond_encoder_class_spec_norm.pkl") },
            checkpoint_path= os.path.join(MODEL_DIR, "final_latent_5_cond_encoder_alpha_20_hidden1024_latentdiffusion_20250723_195439_best")
        ),
        "base_path": os.path.join(DATA_DIR, "analysis_data"),
        "device": "cuda"
    }
    run_preparation_pipeline(pipeline_config)
    run_evaluation_pipeline(pipeline_config)
    run_analysis_pipeline(pipeline_config)

    # analysis_pipeline = AnalysisPipeline(**{"base_path": os.path.join(DATA_DIR, "analysis_data"), "device": "cuda", "saved_model": None})
    # analysis_pipeline.use_hidden_initial_labels = True
    # analysis_pipeline.supported_steps = [AnalysisPipelineSteps.original_geometrics]
    # analysis_pipeline.run()