from s2_fos.tt.interface import Instance, PredictorConfig, Predictor

model_path = './data'

instance = Instance(text_title="A Prototype-Based Few-Shot Named Entity Recognition",
                    abstract="Few-shot Named Entity Recognition (NER) task focuses on identifying name entities on a "
                             "small amount of supervised training data. The work based on prototype network shows "
                             "strong adaptability on the few-shot NER task. We think that the core idea of these "
                             "approaches is to learn how to aggregate the representation of token mappings in vector "
                             "space around entity class. But, as far as we know, no such work has been investigated its"
                             " effect. So, we propose the ClusLoss and the ProEuroLoss aiming to enhance the model's "
                             "ability in terms of aggregating semantic information spatially, thus helping the model "
                             "better distinguish entity types. Experimental results show that ProEuroLoss achieves "
                             "state-of-the-art performance on the average F1 scores for both 1-shot and 5-shot NER "
                             "tasks, while the ClusLoss has competitive performance on such tasks.', "
                             "journal_name='Proceedings of the 8th International Conference on Computing and Artificial "
                             "Intelligence")
config = PredictorConfig()

predictor = Predictor(config, artifacts_dir=model_path)

predictions = predictor.predict_batch([instance])
print(predictions)

