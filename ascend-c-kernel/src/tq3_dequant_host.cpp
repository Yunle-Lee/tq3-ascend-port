
#include "tq3_dequant_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  Tq3DequantTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  int32_t dim_num = x1_shape->GetStorageShape().GetDimNum();
  for (int i = 0; i < dim_num; i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  tiling.set_block_dim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(8);
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  int64_t dim_num = x1_shape->GetDimNum();
  *y_shape = *x1_shape;
  if (dim_num >= 1) {
    int64_t inner_dim = x1_shape->GetDim(dim_num - 1);
    y_shape->SetDim(dim_num - 1, inner_dim * 2);
  }
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
  context->SetOutputDataType(0, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Tq3Dequant : public OpDef {
public:
    explicit Tq3Dequant(const char* name) : OpDef(name)
    {

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(Tq3Dequant);
}
