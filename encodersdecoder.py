from torch import nn
class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def forward(self,X,*args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def init_state(self,en_output,*args):
         raise NotImplementedError
    def forward(self,X,state):
        raise NotImplementedError
class EncoderDecoder(nn.Module):
    def __init__(self, encoder,decoder,**kwargs)
        super().__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,en_X,de_X,*args):
        en_output=self.encoder(en_X,*args)
        de_state=self.decoder.init_state(en_output,*args)
        return self.decoder(de_X,de_state)
