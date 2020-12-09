; ModuleID = "test1.btor2"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%"struct.State" = type {i32, i32}
%"struct.Input" = type {i1}
define void @"init"(%"struct.State"* %".1", %"struct.Input"* %".2", i8* %".3", i64 %".4") 
{
entry:
  %".6" = getelementptr i8, i8* %".3", i64 %".4"
  %".7" = load i8, i8* %".6"
  %".8" = trunc i8 %".7" to i1
  %".9" = add i64 %".4", 1
  %".10" = getelementptr %"struct.Input", %"struct.Input"* %".2", i32 0, i32 0
  store i1 %".8", i1* %".10"
  %".12" = getelementptr %"struct.State", %"struct.State"* %".1", i32 0, i32 0
  store i32 0, i32* %".12"
  %".14" = getelementptr %"struct.State", %"struct.State"* %".1", i32 0, i32 1
  store i32 0, i32* %".14"
  ret void
}

define void @"next"(%"struct.State"* %".1", %"struct.Input"* %".2", i8* %".3", i64 %".4") 
{
entry:
  %".6" = getelementptr %"struct.State", %"struct.State"* %".1", i32 0, i32 0
  %".7" = getelementptr %"struct.State", %"struct.State"* %".1", i32 0, i32 1
  %".8" = load i32, i32* %".6"
  %".9" = load i32, i32* %".7"
  %".10" = getelementptr i8, i8* %".3", i64 %".4"
  %".11" = load i8, i8* %".10"
  %".12" = trunc i8 %".11" to i1
  %".13" = add i64 %".4", 1
  %".14" = getelementptr %"struct.Input", %"struct.Input"* %".2", i32 0, i32 0
  store i1 %".12", i1* %".14"
  %".16" = add i32 %".8", 1
  %".17" = select i1 %".12", i32 %".8", i32 %".16"
  store i32 %".17", i32* %".6"
  %".19" = xor i1 %".12", -1
  %".20" = add i32 %".9", 1
  %".21" = select i1 %".19", i32 %".9", i32 %".20"
  store i32 %".21", i32* %".7"
  ret void
}

define i1 @"bad"(%"struct.State"* %".1", %"struct.Input"* %".2") 
{
entry:
  %".4" = getelementptr %"struct.State", %"struct.State"* %".1", i32 0, i32 0
  %".5" = load i32, i32* %".4"
  %".6" = getelementptr %"struct.State", %"struct.State"* %".1", i32 0, i32 1
  %".7" = load i32, i32* %".6"
  %".8" = getelementptr %"struct.Input", %"struct.Input"* %".2", i32 0, i32 0
  %".9" = load i1, i1* %".8"
  %".10" = icmp eq i32 %".5", 3
  %".11" = icmp eq i32 %".7", 3
  %".12" = and i1 %".10", %".11"
  ret i1 %".12"
}

define i1 @"constraint"(%"struct.State"* %".1", %"struct.Input"* %".2") 
{
entry:
  ret i1 true
}

define i32 @"LLVMFuzzerTestOneInput"(i8* %".1", i64 %".2") 
{
entry:
  %".4" = icmp ult i64 %".2", 1
  br i1 %".4", label %"return.zero", label %"init"
init:
  %".6" = alloca %"struct.State"
  %".7" = alloca %"struct.Input"
  call void @"init"(%"struct.State"* %".6", %"struct.Input"* %".7", i8* %".1", i64 0)
  br label %"for.body"
for.body:
  %".10" = phi i64 [1, %"init"], [%".15", %"next"]
  %".11" = call i1 @"constraint"(%"struct.State"* %".6", %"struct.Input"* %".7")
  br i1 %".11", label %"constraint.true", label %"return.zero"
constraint.true:
  %".13" = call i1 @"bad"(%"struct.State"* %".6", %"struct.Input"* %".7")
  br i1 %".13", label %"error", label %"bad.false"
bad.false:
  %".15" = add i64 %".10", 1
  %".16" = icmp ult i64 %".2", %".15"
  br i1 %".16", label %"return.zero", label %"next"
next:
  call void @"next"(%"struct.State"* %".6", %"struct.Input"* %".7", i8* %".1", i64 %".10")
  br label %"for.body"
return.zero:
  ret i32 0
error:
  call void @"exit"(i32 1)
  unreachable
}

declare void @"exit"(i32 %".1") 
