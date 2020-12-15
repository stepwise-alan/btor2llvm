; ModuleID = ""
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @"main"()
{
.2:
  %".3" = alloca i8, i32 2
  %".4" = getelementptr i8, i8* %".3", i32 0
  store i8 1, i8* %".4"
  %".6" = getelementptr i8, i8* %".3", i32 1
  store i8 2, i8* %".6"
  %".8" = call i32 @"LLVMFuzzerTestOneInput"(i8* %".3", i64 2)
  ret i32 0
}

declare i32 @"LLVMFuzzerTestOneInput"(i8* %".1", i64 %".2")