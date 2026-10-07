"""Recovered execution paths preserve original parameter names and forward semantics.

These adapters are intentionally separate from the composable native models.
Source snapshots and provenance live in recovered/ and reference/.
"""


def build_recovered(config, num_classes):
    if config.backbone == "resnet18":
        from .recovered.resnet import BasicBlock, ResNet, DGCF_Res18

        return (DGCF_Res18 if config.fusion else ResNet)(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes
        )
    from .recovered.convnext import ConvNeXt, DGCF_ConvNeXt

    return (DGCF_ConvNeXt if config.fusion else ConvNeXt)(num_classes=num_classes)


def recovered_features(core, config, image, edge):
    if config.backbone == "resnet18":
        x = core.conv1(image)
        if config.fusion:
            edge = core.edge_branch[:3](edge)
        x = core.conv2_x(x)
        if config.fusion:
            x = core.gate_fusions[0](x, edge)
        x = core.conv3_x(x)
        if config.fusion:
            edge = core.edge_branch[4](edge)
            x = core.gate_fusions[1](x, edge)
        x = core.conv4_x(x)
        if config.fusion:
            edge = core.edge_branch[5](edge)
            x = core.gate_fusions[2](x, edge)
        return core.conv5_x(x)
    x = core.stem(image)
    if config.fusion:
        edge = core.edge_branch[0](edge)
    for index in range(4):
        x = core.stages[index](x)
        if config.fusion:
            x = core.dgf_blocks[index](x, core.dcm_blocks[index](x, edge))
        if index < 3:
            x = core.downsample_layers[index](x)
            if config.fusion:
                edge = core.edge_branch[index + 1](edge)
    return x
