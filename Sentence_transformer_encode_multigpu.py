"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    #Create a large list of 100k sentences
    sentences = ["This is sentence {}".format(i) for i in range(100000)]

    #Define the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    emb = model.encode_multi_process(sentences, pool)
    print("Embeddings computed. Shape:", emb.shape)

    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)



    ###### 추론에서 gpu 사용하기 gpu 1개 사용하게 된다. 


    for test_data in ['/test/rf_hdfs', '/test/labeling_1_1', '/test/labeling_1_2' , '/test/quality_test_1'] ## test data목록, 나중에 추가되면 os로 바꾸자. 
  
        test_data_path = args.data_dir + test_data:
        test_pair = PairData(test_data_path)
        test_samples = test_pair.get_example(shuffle=False, num_data=args.valid_size, gpu=args.gpu)
        test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name=args.data_name)
        logger.debug('This is test prediction result of {}'.format(test_data))
        test_evaluator(sbert_model) 
