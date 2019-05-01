class GANConfig():
    
    def __init__(self):
        
        self.n_epoch = 100
        self.batch_size = 128
        self.lr = 0.0002
        
        self.ckpts_root = 'ckpts/gan/'
        
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
            assert hasattr(self, k), 'Unknown attr '+ k +' !'
            
            setattr(self, k, v)
                
        self.print_config()
        
        return


class ACGANConfig():
    
    def __init__(self):
        
        self.n_epoch = 100
        self.batch_size = 128
        self.lr = 0.0002
        
        self.ckpts_root = 'ckpts/acgan/'
        
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
            assert hasattr(self, k), 'Unknown attr '+ k +' !'
            
            setattr(self, k, v)
                
        self.print_config()
        
        return
    

class DANNConfig():
    
    def __init__(self):
        
        self.n_epoch = 50
        self.batch_size = 512
        self.lr = 0.01
        
        self.ckpts_root = 'ckpts/dann/'
        
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
            assert hasattr(self, k), 'Unknown attr '+ k +' !'
            
            setattr(self, k, v)
                
        self.print_config()
        
        return


class ADDAConfig():
    
    def __init__(self):
        
        self.n_epoch = 50
        self.batch_size = 512
        self.lr = 0.0002
        
        self.ckpts_root = 'ckpts/adda/'
        
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
            assert hasattr(self, k), 'Unknown attr '+ k +' !'
            
            setattr(self, k, v)
                
        self.print_config()
        
        return