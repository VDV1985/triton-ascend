/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferInnerScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace hivm;
using namespace annotation;

using BufferPair = std::pair<Value, Value>;
constexpr int kInlineBufferSize = 8;
using BufferMap = DenseMap<Value, SmallVector<BufferPair, kInlineBufferSize>>;

namespace mlir {
namespace triton {
namespace AddMultiBufferInnerScope {

    struct BlockInfo {
        Value blockId;
        SmallVector<Operation *, 32> ops;
    };

    int getSsbufferId(Operation *op) {
        if (auto idAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
            return idAttr.getInt();
        if (auto idAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.id"))
            return idAttr.getInt();
        return -1;
    }

    void collectNestedOps(Block *block, SmallVector<Operation *, 32> &ops) {
        for (auto &op : *block) {
            ops.push_back(&op);
            for (auto &region : op.getRegions())
                for (auto &innerBlock : region)
                    collectNestedOps(&innerBlock, ops);
        }
    }

    // 获取 forOp 的优先级（数值越小优先级越高）
    static int getForOpPriority(scf::ForOp f) {
        bool hasMainloop = f->hasAttr("ssbuffer.main_loop");
        bool bodyHasMainloop = false;
        bool bodyHasBlockId = false;

        if (auto *term = f.getBody()->getTerminator()) {
            bodyHasMainloop = term->hasAttr("ssbuffer.main_loop");
            bodyHasBlockId = term->getAttrOfType<IntegerAttr>("ssbuffer.block_id") != nullptr;
        }

        bool opHasBlockId = f->getAttrOfType<IntegerAttr>("ssbuffer.block_id") != nullptr;
        bool hasIterArgs = f.getNumResults() > 0 || !f.getInitArgs().empty();

        if (hasMainloop || bodyHasMainloop)
            return 1;
        if (opHasBlockId || bodyHasBlockId)
            return 2;
        if (hasIterArgs)
            return 3;
        return 0;
    }

    scf::ForOp findMainloopInScope(scope::ScopeOp scope) {
        SmallVector<Operation *, 32> allOps;
        collectNestedOps(&scope.getBodyRegion().front(), allOps);

        scf::ForOp mainLoopForOp;
        int bestPriority = INT_MAX;

        for (Operation *op : allOps) {
            auto f = dyn_cast<scf::ForOp>(op);
            if (!f) continue;

            int priority = getForOpPriority(f);
            if (priority > 0 && priority < bestPriority) {
                mainLoopForOp = f;
                bestPriority = priority;
            }
        }
        return mainLoopForOp;
    }

    // 收集单个依赖值到 depValueMap
    static void collectDepValue(Value operand, Block *body, int currentBlockId,
                            DenseMap<Value, int> &outputToBlockId,
                            DenseMap<Value, SmallVector<Value, 8>> &depValueMap,
                            Value groupKey) {
        if (auto barg = dyn_cast<BlockArgument>(operand)) {
            if (barg.getOwner() == body &&
                !llvm::is_contained(depValueMap[groupKey], barg))
                depValueMap[groupKey].push_back(barg);
            return;
        }

        if (outputToBlockId.count(operand) &&
            outputToBlockId[operand] != currentBlockId &&
            !llvm::is_contained(depValueMap[groupKey], operand))
            depValueMap[groupKey].push_back(operand);
    }

    // 递归查找嵌套的 main_loop
    static scf::ForOp findNestedMainloopInForOp(scf::ForOp forOp) {
        SmallVector<Operation *, 32> allOps;
        collectNestedOps(forOp.getBody(), allOps);

        for (Operation *op : allOps) {
            auto nestedFor = dyn_cast<scf::ForOp>(op);
            if (!nestedFor) continue;
            if (nestedFor->hasAttr("ssbuffer.main_loop"))
                return nestedFor;
        }
        return {};
    }

    bool isInsideMainLoopForOp(Operation *op) {
        Operation *parent = op->getParentOp();
        if (!parent) return false;
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            return forOp->hasAttr("ssbuffer.main_loop");
        }
        return false;
    }

    bool isInsideMainLoopForOpTraverse(Operation *op) {
        Operation *parent = op->getParentOp();
        while (parent) {
            if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
                if (forOp->hasAttr("ssbuffer.main_loop")) {
                    return true;
                }
            }
            parent = parent->getParentOp();
        }
        return false;
    }

    DenseMap<Value, SmallVector<Value, 8>>
    collectBlockInfo(scf::ForOp forOp, DenseMap<Value, BlockInfo> &blocks) {
        DenseMap<Value, SmallVector<Value, 8>> depValueMap;
        Block *body = forOp.getBody();
        if (!body) return depValueMap;

        SmallVector<Operation *, 32> allOps;
        collectNestedOps(body, allOps);

        // 1. 收集所有有 ssbuffer.id 的 op 及其结果
        llvm::MapVector<Value, Operation *> opsByValue;
        for (Operation *op : allOps) {
            int id = getSsbufferId(op);
            if (id < 0) continue;
            for (auto res : op->getResults())
                opsByValue[res] = op;
        }
        if (opsByValue.empty()) return depValueMap;

        // 2. 按 ssbuffer id 分组
        llvm::MapVector<int, SmallVector<Operation *, 32>> opsById;
        for (auto &p : opsByValue) {
            int id = getSsbufferId(p.second);
            if (id >= 0) opsById[id].push_back(p.second);
        }

        // 3. 建立输出到 block id 的映射
        DenseMap<Value, int> outputToBlockId;
        for (auto &p : opsById)
            for (Operation *op : p.second)
                for (auto res : op->getResults())
                    outputToBlockId[res] = p.first;

        // 4. 为每个 block 收集依赖值
        for (auto &p : opsById) {
            Value groupKey = p.second.front()->getResult(0);
            BlockInfo bi;
            bi.blockId = groupKey;
            bi.ops = p.second;
            blocks[groupKey] = bi;

            for (Operation *op : bi.ops)
                for (Value operand : op->getOperands())
                    collectDepValue(operand, body, p.first, outputToBlockId, depValueMap, groupKey);
        }

        return depValueMap;
    }

    DenseMap<Value, SmallVector<Operation *, 8>>
    buildDepUserMap(DenseMap<Value, BlockInfo> &blocks) {
        DenseMap<Value, SmallVector<Operation *, 8>> depUserMap;
        for (auto &p : blocks)
            for (Operation *op : p.second.ops)
                for (Value operand : op->getOperands())
                    depUserMap[operand].push_back(op);
        return depUserMap;
    }

    SmallVector<Value, 8>
    collectBufferValues(DenseMap<Value, SmallVector<Value, 8>> &depValueMap) {
        SmallVector<Value, 8> valueList;
        SmallVector<void *, 8> seenPtrs;

        for (auto &p : depValueMap) {
            for (Value depVal : p.second) {
                if (depVal.getDefiningOp() &&
                    !llvm::is_contained(seenPtrs, depVal.getAsOpaquePointer())) {
                    seenPtrs.push_back(depVal.getAsOpaquePointer());

                    auto shapedType = dyn_cast<ShapedType>(depVal.getType());
                    if (!shapedType) continue;

                    valueList.push_back(depVal);
                }
            }
        }

        return valueList;
    }

    SmallVector<Value, 8>
    collectScalarDeps(DenseMap<Value, SmallVector<Value, 8>> &depValueMap,
                    DenseMap<Value, SmallVector<Operation *, 8>> &depUserMap) {
        SmallVector<Value, 8> scalarValueList;

        for (auto &p : depValueMap) {
            for (Value depVal : p.second) {
                if (isa<BlockArgument>(depVal)) continue;

                Operation *depDefinedOp = depVal.getDefiningOp();
                if (!depDefinedOp) continue;

                if (isa<ShapedType>(depVal.getType())) continue;

                auto userIt = depUserMap.find(depVal);
                if (userIt == depUserMap.end()) continue;

                int producerId = getSsbufferId(depDefinedOp);
                if (producerId == -1) continue;

                SmallVector<Operation *, 8> depUsers = userIt->second;
                bool hasCrossBlockUser = false;
                for (Operation *depUser : depUsers) {
                    if (getSsbufferId(depUser) != producerId) {
                        hasCrossBlockUser = true;
                        break;
                    }
                }

                if (hasCrossBlockUser)
                    scalarValueList.push_back(depVal);
            }
        }

        return scalarValueList;
    }

} // namespace AddMultiBufferInnerScope

} // namespace triton
} // namespace mlir

static Value getIterCount(OpBuilder &builder, mlir::scf::ForOp forOp, Location loc,
                          SmallVector<Operation *, 16> *newOps) {
    auto i32Type = builder.getI32Type();
    Value iv = forOp.getInductionVar();
    Value lb = forOp.getLowerBound();
    Value step = forOp.getStep();
    Type ivType = iv.getType();

    bool lbIsZero = false;
    if (Operation *lbDefOp = lb.getDefiningOp()) {
        if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(lbDefOp)) {
            if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                lbIsZero = (intAttr.getInt() == 0);
        }
    }

    Value iterIdx;
    if (lbIsZero) {
        bool stepIsOne = false;
        if (auto constOp = step.getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto intVal = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                stepIsOne = (intVal.getInt() == 1);
        }

        if (stepIsOne) {
            iterIdx = iv;
        } else {
            iterIdx = builder.create<mlir::arith::DivUIOp>(loc, iv, step);
            if (newOps) newOps->push_back(iterIdx.getDefiningOp());
        }
    } else {
        Value diff = builder.create<mlir::arith::SubIOp>(loc, iv, lb);
        iterIdx = builder.create<mlir::arith::DivUIOp>(loc, diff, step);
        if (newOps) {
            newOps->push_back(diff.getDefiningOp());
            newOps->push_back(iterIdx.getDefiningOp());
        }
    }

    if (ivType == i32Type)
        return iterIdx;

    if (ivType.isIndex()) {
        Value result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
        if (newOps) newOps->push_back(result.getDefiningOp());
        return result;
    }

    if (auto intType = dyn_cast<mlir::IntegerType>(ivType)) {
        if (intType.getWidth() < 32) {
            Value result = builder.create<mlir::arith::ExtSIOp>(loc, i32Type, iterIdx);
            if (newOps) newOps->push_back(result.getDefiningOp());
            return result;
        }
        if (intType.getWidth() > 32) {
            Value result = builder.create<mlir::arith::TruncIOp>(loc, i32Type, iterIdx);
            if (newOps) newOps->push_back(result.getDefiningOp());
            return result;
        }
    }

    Value result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
    if (newOps) newOps->push_back(result.getDefiningOp());
    return result;
}

// 创建单个 producer 操作
static Operation *createProducerOp(OpBuilder &builder, Location loc, Value src, Value dst) {
    return builder.create<mlir::bufferization::MaterializeInDestinationOp>(
        loc, mlir::Type{}, src, dst,
        mlir::UnitAttr::get(builder.getContext()),
        mlir::UnitAttr::get(builder.getContext()));
}

// 构建 producer 的 if 链
static void buildProducerIfChain(OpBuilder &builder, Location loc, Value bufIdx,
                                Value depVal, SmallVector<BufferPair, kInlineBufferSize> &buffers,
                                SmallVector<Operation *, 16> &newOps) {
    int N = buffers.size();
    Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    Value firstCond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, bufIdx, zero);
    auto firstIf = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, firstCond, true, true);

    newOps.push_back(zero.getDefiningOp());
    newOps.push_back(firstCond.getDefiningOp());
    newOps.push_back(firstIf);

    builder.setInsertionPointToStart(&firstIf.getThenRegion().front());
    newOps.push_back(createProducerOp(builder, loc, depVal, buffers[0].second));
    builder.create<mlir::scf::YieldOp>(loc);

    mlir::Block *currentElseBlock = &firstIf.getElseRegion().front();
    for (int i = 1; i < N - 1; ++i) {
        builder.setInsertionPointToStart(currentElseBlock);
        Value iVal = builder.create<mlir::arith::ConstantIntOp>(loc, i, 32);
        Value cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, bufIdx, iVal);
        auto nestedIf = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond, true, true);

        newOps.push_back(iVal.getDefiningOp());
        newOps.push_back(cond.getDefiningOp());
        newOps.push_back(nestedIf);

        builder.setInsertionPointToStart(&nestedIf.getThenRegion().front());
        newOps.push_back(createProducerOp(builder, loc, depVal, buffers[i].second));
        builder.create<mlir::scf::YieldOp>(loc);

        currentElseBlock = &nestedIf.getElseRegion().front();
    }

    builder.setInsertionPointToStart(currentElseBlock);
    newOps.push_back(createProducerOp(builder, loc, depVal, buffers[N - 1].second));
    builder.create<mlir::scf::YieldOp>(loc);

    builder.setInsertionPointAfter(firstIf);
}

static SmallVector<Operation *, 16> insertProducerLogic(OpBuilder &builder, Value depVal,
                                                       SmallVector<BufferPair, kInlineBufferSize> &buffers,
                                                       mlir::scf::ForOp forOp) {
    SmallVector<Operation *, 16> newOps;
    int N = buffers.size();
    Location loc = depVal.getLoc();

    Value iterCount = getIterCount(builder, forOp, loc, &newOps);

    if (N == 1) {
        newOps.push_back(createProducerOp(builder, loc, depVal, buffers[0].second));
        return newOps;
    }

    Value Nval = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
    Value bufIdx = builder.create<mlir::arith::RemSIOp>(loc, iterCount, Nval);
    newOps.push_back(Nval.getDefiningOp());
    newOps.push_back(bufIdx.getDefiningOp());

    buildProducerIfChain(builder, loc, bufIdx, depVal, buffers, newOps);

    return newOps;
}

// N==1 时 consumer 的处理（直接返回 buffer）
static Operation *handleSingleBufferConsumer(OpBuilder &builder, Location loc,
                                             SmallVector<BufferPair, kInlineBufferSize> &buffers) {
    auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
    auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
    return builder.create<mlir::bufferization::ToTensorOp>(
        loc, tensorType, buffers[0].second,
        mlir::UnitAttr::get(builder.getContext()),
        mlir::UnitAttr::get(builder.getContext()));
}

// 构建 consumer 的 if 链（tensor 类型）
static void buildConsumerTensorIfChain(OpBuilder &builder, Location loc, Value readIdx,
                                      SmallVector<BufferPair, kInlineBufferSize> &buffers,
                                      SmallVector<Operation *, 16> &newOps,
                                      SmallVector<Operation *, 4> &outIfOps,
                                      int groupId) {
    int N = buffers.size();
    auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
    auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
    SmallVector<mlir::Type> resultTypes{tensorType};

    Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    Value firstCond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, readIdx, zero);

    auto firstIf = builder.create<mlir::scf::IfOp>(loc, resultTypes, firstCond, true, true);
    if (groupId >= 0) {
        firstIf->setAttr("ssbuffer.intraDeps", builder.getI32ArrayAttr({groupId, 0}));
    }
    newOps.push_back(zero.getDefiningOp());
    newOps.push_back(firstCond.getDefiningOp());
    newOps.push_back(firstIf);
    outIfOps.push_back(firstIf);

    builder.setInsertionPointToStart(&firstIf.getThenRegion().front());
    auto ti0 = builder.create<mlir::bufferization::ToTensorOp>(
        loc, tensorType, buffers[0].second,
        mlir::UnitAttr::get(builder.getContext()),
        mlir::UnitAttr::get(builder.getContext()));
    newOps.push_back(ti0);
    builder.create<mlir::scf::YieldOp>(loc, ti0.getResult());

    mlir::Block *currentElseBlock = &firstIf.getElseRegion().front();
    for (int i = 1; i < N - 1; ++i) {
        builder.setInsertionPointToStart(currentElseBlock);
        Value iVal = builder.create<mlir::arith::ConstantIntOp>(loc, i, 32);
        Value cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, readIdx, iVal);
        auto nestedIf = builder.create<mlir::scf::IfOp>(loc, resultTypes, cond, true, true);

        newOps.push_back(iVal.getDefiningOp());
        newOps.push_back(cond.getDefiningOp());
        newOps.push_back(nestedIf);
        outIfOps.push_back(nestedIf);

        builder.setInsertionPointToStart(&nestedIf.getThenRegion().front());
        auto ti = builder.create<mlir::bufferization::ToTensorOp>(
            loc, tensorType, buffers[i].second,
            mlir::UnitAttr::get(builder.getContext()),
            mlir::UnitAttr::get(builder.getContext()));
        newOps.push_back(ti);
        builder.create<mlir::scf::YieldOp>(loc, ti.getResult());

        currentElseBlock = &nestedIf.getElseRegion().front();
    }

    builder.setInsertionPointToStart(currentElseBlock);
    auto tiLast = builder.create<mlir::bufferization::ToTensorOp>(
        loc, tensorType, buffers[N - 1].second,
        mlir::UnitAttr::get(builder.getContext()),
        mlir::UnitAttr::get(builder.getContext()));
    newOps.push_back(tiLast);
    builder.create<mlir::scf::YieldOp>(loc, tiLast.getResult());

    builder.setInsertionPointAfter(firstIf);
}

static SmallVector<Operation *, 16> insertConsumerLogic(OpBuilder &builder,
                                                       Value depVal,
                                                       SmallVector<BufferPair, kInlineBufferSize> &buffers,
                                                       mlir::scf::ForOp forOp,
                                                       SmallVector<Operation *, 4> &outIfOps,
                                                       int groupId = -1) {
    SmallVector<Operation *, 16> newOps;
    int N = buffers.size();
    Location loc = builder.getInsertionPoint()->getLoc();

    Value iterCount = getIterCount(builder, forOp, loc, &newOps);

    Value Nval = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
    Value readIdx = builder.create<mlir::arith::RemSIOp>(loc, iterCount, Nval);
    newOps.push_back(Nval.getDefiningOp());
    newOps.push_back(readIdx.getDefiningOp());

    if (N == 1) {
        outIfOps.push_back(handleSingleBufferConsumer(builder, loc, buffers));
        return newOps;
    }

    buildConsumerTensorIfChain(builder, loc, readIdx, buffers, newOps, outIfOps, groupId);

    return newOps;
}

static void addBlockAttrForOps(SmallVector<Operation *, 16> &newOps, int blockId,
                              OpBuilder &builder) {
    auto attr = builder.getI32IntegerAttr(blockId);
    for (auto *op : newOps)
        op->setAttr("ssbuffer.block_id", attr);
}

// 添加 dep_mark 属性到操作
static void addDepMarkAttr(Operation *op, int depMark, OpBuilder &builder) {
    if (auto existingAttr = op->getAttrOfType<mlir::ArrayAttr>("ssbuffer.dep_mark")) {
        SmallVector<int> marks;
        for (auto attr : existingAttr)
            marks.push_back(cast<mlir::IntegerAttr>(attr).getInt());
        marks.push_back(depMark);
        op->setAttr("ssbuffer.dep_mark", builder.getI32ArrayAttr(marks));
    } else {
        op->setAttr("ssbuffer.dep_mark", builder.getI32ArrayAttr({depMark}));
    }
}

// 收集跨 block 的用户操作
static SmallVector<Operation *, 8> collectCrossBlockUsers(
    Value depVal, int producerId,
    DenseMap<Value, SmallVector<Operation *, 8>> &depUserMap) {
    SmallVector<Operation *, 8> crossBlockUsers;

    auto userIt = depUserMap.find(depVal);
    if (userIt == depUserMap.end()) return crossBlockUsers;

    for (Operation *depUser : userIt->second) {
        if (triton::AddMultiBufferInnerScope::getSsbufferId(depUser) != producerId &&
            triton::AddMultiBufferInnerScope::isInsideMainLoopForOpTraverse(depUser))
            crossBlockUsers.push_back(depUser);
    }
    return crossBlockUsers;
}

static void markScalarDeps(SmallVector<Value, 8> &scalarValueList,
                           DenseMap<Value, SmallVector<Operation *, 8>> &depUserMap,
                           OpBuilder &builder) {
    static int nextDepMark = 1;

    for (Value depVal : scalarValueList) {
        Operation *depDefinedOp = depVal.getDefiningOp();
        if (!depDefinedOp) continue;

        if (!triton::AddMultiBufferInnerScope::isInsideMainLoopForOp(depDefinedOp)) continue;

        int producerId = triton::AddMultiBufferInnerScope::getSsbufferId(depDefinedOp);
        auto crossBlockUsers = collectCrossBlockUsers(depVal, producerId, depUserMap);

        if (crossBlockUsers.empty()) continue;

        int depMark = nextDepMark++;
        addDepMarkAttr(depDefinedOp, depMark, builder);

        for (Operation *depUser : crossBlockUsers) {
            addDepMarkAttr(depUser, depMark, builder);
        }
    }
}


void mlir::triton::AddMultiBufferInnerScopePass::runOnOperation()
{
    ModuleOp module = getOperation();
    OpPassManager pm(module.getOperationName());
    LDBG("Enter pass.");

    // Step 1: Walk scope operations

    // Step 2: Find main loop in each scope

    // Step 3: Apply inner multi-buffer optimization

    if (failed(runPipeline(pm, module))) {
        module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
        signalPassFailure();
    }

    LDBG("Process successfully");
}

void mlir::triton::AddMultiBufferInnerScopePass::getDependentDialects(
    DialectRegistry &registry) const
{
    registry.insert<scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect,
                    linalg::LinalgDialect, bufferization::BufferizationDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createAddMultiBufferInnerScopePass()
{
    return std::make_unique<AddMultiBufferInnerScopePass>();
}