function lgraph = addSelfAttentionLayer(lgraph, layerName, nextLayerName, numChannels)
    % Debug print to verify input
    disp(">> Self-Attention: Projecting to " + num2str(numChannels) + " channels");

    % Define Self-Attention Layers
    attentionConvQLayer = convolution2dLayer(1, numChannels, 'Name', 'attention_convQ', 'WeightsInitializer', 'glorot');
    attentionConvKLayer = convolution2dLayer(1, numChannels, 'Name', 'attention_convK', 'WeightsInitializer', 'glorot');
    attentionConvVLayer = convolution2dLayer(1, numChannels, 'Name', 'attention_convV', 'WeightsInitializer', 'glorot');
    attentionSoftmaxLayer = softmaxLayer('Name', 'attention_softmax');

    % Multiplication and addition layers
    softmaxMultLayer = multiplicationLayer(2, 'Name', 'softmax_mult');
    attentionAddLayer = additionLayer(2, 'Name', 'attention_add');

    % Add attention layers
    lgraph = addLayers(lgraph, attentionConvQLayer);
    lgraph = addLayers(lgraph, attentionConvKLayer);
    lgraph = addLayers(lgraph, attentionConvVLayer);
    lgraph = addLayers(lgraph, attentionSoftmaxLayer);
    lgraph = addLayers(lgraph, softmaxMultLayer);
    lgraph = addLayers(lgraph, attentionAddLayer);

    % Connect Q, K, V
    lgraph = connectLayers(lgraph, layerName, 'attention_convQ');
    lgraph = connectLayers(lgraph, layerName, 'attention_convK');
    lgraph = connectLayers(lgraph, layerName, 'attention_convV');

    % Connect K to softmax
    lgraph = connectLayers(lgraph, 'attention_convK', 'attention_softmax');

    % Multiply softmax output with V
    lgraph = connectLayers(lgraph, 'attention_softmax', 'softmax_mult/in1');
    lgraph = connectLayers(lgraph, 'attention_convV', 'softmax_mult/in2');

    % Add softmax result to Q
    lgraph = connectLayers(lgraph, 'attention_convQ', 'attention_add/in1');
    lgraph = connectLayers(lgraph, 'softmax_mult', 'attention_add/in2');

    % Disconnect original connection
    lgraph = disconnectLayers(lgraph, layerName, nextLayerName);

    % Adjust the attention output to match the original channels (128)
    adjustedAttentionLayer = convolution2dLayer(1, numChannels, ...
        'Name', 'adjusted_attention', 'WeightsInitializer', 'glorot');
    lgraph = addLayers(lgraph, adjustedAttentionLayer);
    lgraph = connectLayers(lgraph, 'attention_add', 'adjusted_attention');

    % Merge with original feature path
    mergeAddLayer = additionLayer(2, 'Name', 'merge_add');
    lgraph = addLayers(lgraph, mergeAddLayer);
    lgraph = connectLayers(lgraph, layerName, 'merge_add/in1');
    lgraph = connectLayers(lgraph, 'adjusted_attention', 'merge_add/in2');

    % Connect to next layer
    lgraph = connectLayers(lgraph, 'merge_add', nextLayerName);
end
