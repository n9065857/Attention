function lgraph = addSelfAttentionLayer(lgraph, layerName, nextLayerName, numChannels)
    % Define Self-Attention Layers
    attentionConvQLayer = convolution2dLayer(1, numChannels, 'Name', 'attention_convQ', 'WeightsInitializer', 'glorot');
    attentionConvKLayer = convolution2dLayer(1, numChannels, 'Name', 'attention_convK', 'WeightsInitializer', 'glorot');
    attentionConvVLayer = convolution2dLayer(1, numChannels, 'Name', 'attention_convV', 'WeightsInitializer', 'glorot');
    attentionSoftmaxLayer = softmaxLayer('Name', 'attention_softmax');
    
    % Multiplication layer to combine softmax output with V path
    softmaxMultLayer = multiplicationLayer(2, 'Name', 'softmax_mult');
    
    % Addition layer to combine Q and the result of softmax multiplication
    attentionAddLayer = additionLayer(2, 'Name', 'attention_add');
    
    % Add the attention layers to the graph
    lgraph = addLayers(lgraph, attentionConvQLayer);
    lgraph = addLayers(lgraph, attentionConvKLayer);
    lgraph = addLayers(lgraph, attentionConvVLayer);
    lgraph = addLayers(lgraph, attentionSoftmaxLayer);
    lgraph = addLayers(lgraph, softmaxMultLayer);
    lgraph = addLayers(lgraph, attentionAddLayer);

    % Connect the original layer to the Q, K, and V convolutional layers
    lgraph = connectLayers(lgraph, layerName, 'attention_convQ');
    lgraph = connectLayers(lgraph, layerName, 'attention_convK');
    lgraph = connectLayers(lgraph, layerName, 'attention_convV');

    % Connect the K path to the softmax layer
    lgraph = connectLayers(lgraph, 'attention_convK', 'attention_softmax');
    
    % Multiply the softmax output with the V path
    lgraph = connectLayers(lgraph, 'attention_softmax', 'softmax_mult/in1');
    lgraph = connectLayers(lgraph, 'attention_convV', 'softmax_mult/in2');

    % Add the result of the softmax multiplication to the Q path
    lgraph = connectLayers(lgraph, 'attention_convQ', 'attention_add/in1');
    lgraph = connectLayers(lgraph, 'softmax_mult', 'attention_add/in2');

    % Disconnect the original connection to avoid conflict
    lgraph = disconnectLayers(lgraph, layerName, nextLayerName);

    % Adjust the number of channels from 'attention_add' output to match the original path (112 channels)
    adjustedAttentionLayer = convolution2dLayer(1, 728, ...
                                                'Name', 'adjusted_attention', ...
                                                'WeightsInitializer', 'glorot');
    lgraph = addLayers(lgraph, adjustedAttentionLayer);
    lgraph = connectLayers(lgraph, 'attention_add', 'adjusted_attention');

    % Add an Addition Layer to combine original and attention paths
    mergeAddLayer = additionLayer(2, 'Name', 'merge_add');
    lgraph = addLayers(lgraph, mergeAddLayer);

    % Connect the original path and adjusted attention output to the merge layer
    lgraph = connectLayers(lgraph, layerName, 'merge_add/in1');
    lgraph = connectLayers(lgraph, 'adjusted_attention', 'merge_add/in2');

    % Connect the merged output to the next layer
    lgraph = connectLayers(lgraph, 'merge_add', nextLayerName);
end

