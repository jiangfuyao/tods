from d3m import container
from tods.feature_analysis import DiscreteCosineTransform

class DiscreteCosineTransformSK():
    def __init__(self, use_semantic_types=True, use_columns=(0,), return_result='append'):
        
        hyperparams_class = DiscreteCosineTransform.DiscreteCosineTransform.metadata.get_hyperparams()
        hp = hyperparams_class.defaults().replace({
            'use_semantic_types':use_semantic_types,
            'use_columns': use_columns,
            'return_result': return_result,
        })

        self.primitive = DiscreteCosineTransform.DiscreteCosineTransform(hyperparams=hp)

    def transform(self, X):
        inputs = {}
        for i in range(len(X)):
            inputs['col_'+str(i)] = list(X[i])
        inputs = container.DataFrame(inputs, columns=list(inputs.keys()), generate_metadata=True)

        outputs = self.primitive.produce(inputs=inputs).value.to_numpy()

        return outputs

if __name__ == '__main__':
    import numpy as np
    transformer = DiscreteCosineTransformSK()
    data = np.array([[1,2,3],[4,5,6]])
    data = transformer.transform(data)
    print(data)

