from s2_fos import S2FOS

if __name__ == "__main__":

    model_path = './data'

    papers = [{'title': "A Prototype-Based Few-Shot Named Entity Recognition",
               'abstract': "Few-shot Named Entity Recognition (NER) task focuses on identifying name entities on a "
                                "small amount of supervised training data. The work based on prototype network shows "
                                "strong adaptability on the few-shot NER task. We think that the core idea of these "
                                "approaches is to learn how to aggregate the representation of token mappings in vector "
                                "space around entity class. But, as far as we know, no such work has been investigated its"
                                " effect. So, we propose the ClusLoss and the ProEuroLoss aiming to enhance the model's "
                                "ability in terms of aggregating semantic information spatially, thus helping the model "
                                "better distinguish entity types. Experimental results show that ProEuroLoss achieves "
                                "state-of-the-art performance on the average F1 scores for both 1-shot and 5-shot NER "
                                "tasks, while the ClusLoss has competitive performance on such tasks.",
                'journal_name': "Proceedings of the 8th International Conference on Computing and Artificial Intelligence",
               'venue_name': ''
               }]

    predictor = S2FOS(data_dir=model_path)

    print(f'Predictions for the papers {papers} \n'
          f'\n ************* \n'
          f'{predictor.predict(papers)}')

    print('Decision function: \n')
    print(predictor.decision_function(papers))
