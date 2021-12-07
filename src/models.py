import torch.nn as nn
import torch.nn.functional as F


class FeedForward( nn.Module ):
    def __init__( self ):
        super( FeedForward, self ).__init__()
        self.fc1 = nn.Linear( 100, 10 )
        self.fc2 = nn.Linear( 10, 1 )

    # __init__

    def forward( self, x ):
        x = self.fc1( x )
        x = F.relu( x )
        x = self.fc2( x )

        return x

    # forward
# class: FeedForward
