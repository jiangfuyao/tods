from d3m import container
from tods.feature_analysis import DiscreteCosineTransform

class InterfaceSK():
    def __init__(self,hyperparams_class,use_semantic_types=True, use_columns=(0,), return_result='append'):
        self.hyperparams_class=hyperparams_class
       
    def transform(self, X):
        inputs = {}
        for i in range(len(X)):
            inputs['col_'+str(i)] = list(X[i])
        inputs = container.DataFrame(inputs, columns=list(inputs.keys()), generate_metadata=True)

        outputs = self.primitive.produce(inputs=inputs).value.to_numpy()

        return outputs




class DiscreteCosineTransformSK(InterfaceSK):
    def __init__(self,hyperparams_class,use_semantic_types=True, use_columns=(0,), return_result='append'):
        super().__init__(hyperparams_class,use_semantic_types=True, use_columns=(0,), return_result='append')
        hyperparams_class = DiscreteCosineTransform.DiscreteCosineTransform.metadata.get_hyperparams()
        hp = hyperparams_class.defaults().replace({
            'use_semantic_types':use_semantic_types,
            'use_columns': use_columns,
            'return_result': return_result,
        })
        self.primitive = DiscreteCosineTransform.DiscreteCosineTransform(hyperparams=hp)






if __name__ == '__main__':
    import numpy as np
    transformer = DiscreteCosineTransformSK(InterfaceSK)
    data = np.array([[1,2,3],[4,5,6]])
    data = transformer.transform(data)
    print(data)



   
