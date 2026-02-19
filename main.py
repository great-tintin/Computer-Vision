diff --git a/eoir_backbone.py b/eoir_backbone.py
new file mode 100644
index 0000000000000000000000000000000000000000..21f977788c44cfa81083d10931b7dea0922d7a10
--- /dev/null
+++ b/eoir_backbone.py
@@ -0,0 +1,106 @@
+import torch
+import torch.nn as nn
+from torchvision import models
+from torchvision.models import ResNet34_Weights
+
+
+class ResNet34Trunk(nn.Module):
+    """ResNet-34 trunk up to layer3 + AdaptiveAvgPool2d((16, 16))."""
+
+    def __init__(self, weights: ResNet34_Weights | None, in_channels: int = 3) -> None:
+        super().__init__()
+        if in_channels not in (1, 3):
+            raise ValueError(f"in_channels must be 1 or 3, got {in_channels}.")
+
+        # NOTE: resnet34 is the official torchvision factory function,
+        # not a custom implementation in this file.
+        backbone = models.resnet34(weights=weights)
+
+        # IR single-channel handling: replace conv1 and optionally initialize from averaged pretrained weights.
+        if in_channels == 1:
+            old_conv1 = backbone.conv1
+            new_conv1 = nn.Conv2d(
+                in_channels=1,
+                out_channels=old_conv1.out_channels,
+                kernel_size=old_conv1.kernel_size,
+                stride=old_conv1.stride,
+                padding=old_conv1.padding,
+                bias=old_conv1.bias is not None,
+            )
+            if weights is not None:
+                with torch.no_grad():
+                    new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
+            backbone.conv1 = new_conv1
+
+        self.conv1 = backbone.conv1
+        self.bn1 = backbone.bn1
+        self.relu = backbone.relu
+        self.maxpool = backbone.maxpool
+        self.layer1 = backbone.layer1
+        self.layer2 = backbone.layer2
+        self.layer3 = backbone.layer3
+        self.out_pool = nn.AdaptiveAvgPool2d((16, 16))
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        x = self.conv1(x)
+        x = self.bn1(x)
+        x = self.relu(x)
+        x = self.maxpool(x)
+        x = self.layer1(x)
+        x = self.layer2(x)
+        x = self.layer3(x)
+        x = self.out_pool(x)
+        return x
+
+
+class EOIRBackbone(nn.Module):
+    """
+    EO/IR dual backbone with independent ResNet-34 trunks (no shared weights).
+
+    Args:
+        c_ir: IR input channel count (must be 1 or 3).
+        weights_eo: torchvision.models.ResNet34_Weights.DEFAULT or None.
+        weights_ir: torchvision.models.ResNet34_Weights.DEFAULT or None.
+    """
+
+    def __init__(
+        self,
+        c_ir: int,
+        weights_eo: ResNet34_Weights | None = ResNet34_Weights.DEFAULT,
+        weights_ir: ResNet34_Weights | None = ResNet34_Weights.DEFAULT,
+    ) -> None:
+        super().__init__()
+        if c_ir not in (1, 3):
+            raise ValueError(f"C_ir must be 1 or 3, got {c_ir}.")
+
+        self.eo_backbone = ResNet34Trunk(weights=weights_eo, in_channels=3)
+        self.ir_backbone = ResNet34Trunk(weights=weights_ir, in_channels=c_ir)
+
+    def forward(self, eo: torch.Tensor, ir: torch.Tensor) -> dict[str, torch.Tensor]:
+        b = eo.shape[0]
+        eo_fmap = self.eo_backbone(eo)
+        ir_fmap = self.ir_backbone(ir)
+
+        assert list(eo_fmap.shape) == [b, 256, 16, 16], (
+            f"Expected eo_fmap shape [B,256,16,16], got {list(eo_fmap.shape)}"
+        )
+        assert list(ir_fmap.shape) == [b, 256, 16, 16], (
+            f"Expected ir_fmap shape [B,256,16,16], got {list(ir_fmap.shape)}"
+        )
+
+        return {"eo_fmap": eo_fmap, "ir_fmap": ir_fmap}
+
+
+if __name__ == "__main__":
+    # 1) Test with IR single-channel input
+    model_ir1 = EOIRBackbone(c_ir=1, weights_eo=None, weights_ir=None)
+    eo = torch.randn(2, 3, 256, 256)
+    ir1 = torch.randn(2, 1, 256, 256)
+    out1 = model_ir1(eo, ir1)
+    print("[Test 1] eo_fmap:", out1["eo_fmap"].shape, ", ir_fmap:", out1["ir_fmap"].shape)
+
+    # 2) Test with IR three-channel input
+    model_ir3 = EOIRBackbone(c_ir=3, weights_eo=None, weights_ir=None)
+    ir3 = torch.randn(2, 3, 256, 256)
+    out2 = model_ir3(eo, ir3)
+    print("[Test 2] eo_fmap:", out2["eo_fmap"].shape, ", ir_fmap:", out2["ir_fmap"].shape)
