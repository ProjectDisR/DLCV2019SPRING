class BaselineConfig():
    
    def __init__(self):
        
        self.env = 'YOLOv1_Baseline'
        self.port = 8888
        
        self.batch_size = 25
        self.n_epoch = 20000
        self.lr = 0.001
        
        self.thres = 0.01
        
        self.ckpts_root = 'ckpts/baseline/'
        self.predicts_root = 'predictTxt/baseline/'
        
        return
        
    def print_config(self):
        
        print('\n')
        
        import inspect
        
        for k in dir(self):   
            if not k.startswith('__') and not inspect.ismethod(getattr(self, k)):
                print('   ', k, ':', getattr(self, k))
                
        return
    
    def parse(self, kwargs):
        
        for k, v in kwargs.items():
            
            if not hasattr(self, k):
                raise Exception('Unknown attr '+ k +' !')
            else:
                setattr(self, k, v)
                
        self.print_config()
        
        return
    
class BestConfig():
    
    def __init__(self):
        
        self.env = 'YOLOv1_Best'
        self.port = 8888
        
        self.n_epoch = 20000
        self.batch_size = 25
        self.lr = 0.001
        
        self.thres = 0.01
        
        self.ckpts_root = 'ckpts/best/'
        self.predicts_root = 'predictTxt/best/'
        
        return
        
    def print_config(self):
        
        print('\n')
        
        import inspect
        
        for k in dir(self):   
            if not k.startswith('__') and not inspect.ismethod(getattr(self, k)):
                print('   ', k, ':', getattr(self, k))
                
        return
    
    def parse(self, kwargs):
        
        for k, v in kwargs.items():
            
            if not hasattr(self, k):
                raise Exception('Unknown attr '+ k +' !')
            else:
                setattr(self, k, v)
                
        self.print_config()
        
        return