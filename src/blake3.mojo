"""BLAKE3 tree hashing, keyed hashing, key derivation, and XOF output."""

from std.memory import alloc
from std.memory import unsafe_memcpy
from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys.info import simd_width_of


comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]

comptime CHUNK_START: UInt32 = 1
comptime CHUNK_END: UInt32 = 2
comptime PARENT: UInt32 = 4
comptime ROOT: UInt32 = 8
comptime PARALLEL_THRESHOLD = 4 * 1024 * 1024


@always_inline
def sync_parallelize[FuncType: def(Int) -> None](
    func: FuncType, count: Int
):
    @__parameter
    @always_inline
    def wrapped(i: Int):
        func(i)

    @always_inline
    @__parameter
    async def task_fn(i: Int):
        wrapped(i)

    var tasks = TaskGroup()
    for i in range(count):
        tasks.create_task(task_fn(i))
    tasks.wait()


@always_inline
def parallelize[
    origins: OriginSet,
    //,
    func: def(Int) capturing[origins] -> None,
](num_work_items: Int, num_workers: Int):
    def unified_func(i: Int):
        func(i)

    var chunk_size, extra_items = divmod(num_work_items, num_workers)

    @always_inline
    def worker(worker_index: Int) {imm chunk_size, imm extra_items}:
        var start = worker_index * chunk_size + min(worker_index, extra_items)
        for i in range(chunk_size + Int(worker_index < extra_items)):
            unified_func(start + i)

    sync_parallelize(worker, num_workers)


@always_inline
def iv(i: Int) -> UInt32:
    if i == 0:
        return 0x6A09E667
    if i == 1:
        return 0xBB67AE85
    if i == 2:
        return 0x3C6EF372
    if i == 3:
        return 0xA54FF53A
    if i == 4:
        return 0x510E527F
    if i == 5:
        return 0x9B05688C
    if i == 6:
        return 0x1F83D9AB
    return 0x5BE0CD19


@always_inline
def rotr32x[W: Int](
    value: SIMD[DType.uint32, W], amount: Int
) -> SIMD[DType.uint32, W]:
    var right = SIMD[DType.uint32, W](UInt32(amount))
    var left = SIMD[DType.uint32, W](UInt32(32 - amount))
    return (value >> right) | (value << left)


@always_inline
def mix4[W: Int](
    mut a: SIMD[DType.uint32, W],
    mut b: SIMD[DType.uint32, W],
    mut c: SIMD[DType.uint32, W],
    mut d: SIMD[DType.uint32, W],
    mx: SIMD[DType.uint32, W],
    my: SIMD[DType.uint32, W],
):
    a = a + b + mx
    d = rotr32x(d ^ a, 16)
    c = c + d
    b = rotr32x(b ^ c, 12)
    a = a + b + my
    d = rotr32x(d ^ a, 8)
    c = c + d
    b = rotr32x(b ^ c, 7)


@always_inline
def message4[i0: Int, i1: Int, i2: Int, i3: Int](
    block: Array[UInt32, 16],
) -> SIMD[DType.uint32, simd_width_of[DType.float64]()] :
    comptime W = simd_width_of[DType.float64]()
    return SIMD[DType.uint32, W](
        block[i0], block[i1], block[i2], block[i3]
    )


@always_inline
def round4[
    i0: Int,
    i1: Int,
    i2: Int,
    i3: Int,
    i4: Int,
    i5: Int,
    i6: Int,
    i7: Int,
    i8: Int,
    i9: Int,
    i10: Int,
    i11: Int,
    i12: Int,
    i13: Int,
    i14: Int,
    i15: Int,
](
    mut a: SIMD[DType.uint32, simd_width_of[DType.float64]()],
    mut b: SIMD[DType.uint32, simd_width_of[DType.float64]()],
    mut c: SIMD[DType.uint32, simd_width_of[DType.float64]()],
    mut d: SIMD[DType.uint32, simd_width_of[DType.float64]()],
    block: Array[UInt32, 16],
):
    mix4(
        a,
        b,
        c,
        d,
        message4[i0, i2, i4, i6](block),
        message4[i1, i3, i5, i7](block),
    )
    b = b.shuffle[1, 2, 3, 0]()
    c = c.shuffle[2, 3, 0, 1]()
    d = d.shuffle[3, 0, 1, 2]()
    mix4(
        a,
        b,
        c,
        d,
        message4[i8, i10, i12, i14](block),
        message4[i9, i11, i13, i15](block),
    )
    b = b.shuffle[3, 0, 1, 2]()
    c = c.shuffle[2, 3, 0, 1]()
    d = d.shuffle[1, 2, 3, 0]()


def compress(
    cv: Array[UInt32, 8],
    block: Array[UInt32, 16],
    counter: UInt64,
    block_len: Int,
    flags: UInt32,
    mut result: Array[UInt32, 16],
):
    comptime W = simd_width_of[DType.float64]()
    var cv_pointer = cv.unsafe_ptr()
    var a = cv_pointer.load[width=W](0)
    var b = cv_pointer.load[width=W](W)
    var c = SIMD[DType.uint32, W](
        iv(0), iv(1), iv(2), iv(3)
    )
    var d = SIMD[DType.uint32, W](
        UInt32(counter),
        UInt32(counter >> 32),
        UInt32(block_len),
        flags,
    )

    round4[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15](
        a, b, c, d, block
    )
    round4[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8](
        a, b, c, d, block
    )
    round4[3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1](
        a, b, c, d, block
    )
    round4[10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6](
        a, b, c, d, block
    )
    round4[12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4](
        a, b, c, d, block
    )
    round4[9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7](
        a, b, c, d, block
    )
    round4[11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13](
        a, b, c, d, block
    )

    var result_pointer = result.unsafe_ptr()
    result_pointer.store(0, a ^ c)
    result_pointer.store(W, b ^ d)
    result_pointer.store(2 * W, c ^ cv_pointer.load[width=W](0))
    result_pointer.store(3 * W, d ^ cv_pointer.load[width=W](W))


def load_block(
    data: BPtr,
    offset: Int,
    size: Int,
    mut block: Array[UInt32, 16],
):
    comptime W = simd_width_of[DType.float64]()
    var block_pointer = block.unsafe_ptr()
    var zero = SIMD[DType.uint32, W](0)
    var i = 0
    while i + W <= 16:
        block_pointer.store(i, zero)
        i += W
    while i < 16:
        block_pointer[i] = 0
        i += 1
    var full_words = size // 4
    var source = (data + offset).bitcast[UInt32]()
    i = 0
    while i + W <= full_words:
        var values = source.load[width=W, alignment=1](i)
        block_pointer.store(i, values)
        i += W
    while i < full_words:
        block_pointer[i] = source.load[alignment=1](i)
        i += 1
    var byte_index = full_words * 4
    while byte_index < size:
        var word = byte_index // 4
        var shift = (byte_index % 4) * 8
        block_pointer[word] |= (
            UInt32(data[offset + byte_index]) << UInt32(shift)
        )
        byte_index += 1


@always_inline
def copy8(
    source: Array[UInt32, 8],
    mut destination: Array[UInt32, 8],
):
    comptime W = simd_width_of[DType.float64]()
    var source_pointer = source.unsafe_ptr()
    var destination_pointer = destination.unsafe_ptr()
    var i = 0
    while i + W <= 8:
        destination_pointer.store(
            i, source_pointer.load[width=W](i)
        )
        i += W
    while i < 8:
        destination_pointer[i] = source_pointer[i]
        i += 1


def chunk_output(
    data: BPtr,
    offset: Int,
    size: Int,
    chunk_counter: UInt64,
    key: Array[UInt32, 8],
    flags: UInt32,
    mut output_cv: Array[UInt32, 8],
    mut output_block: Array[UInt32, 16],
) -> Tuple[Int, UInt32]:
    var cv = Array[UInt32, 8](fill=0)
    var block = Array[UInt32, 16](fill=0)
    var compressed = Array[UInt32, 16](fill=0)
    copy8(key, cv)

    var complete_before_last = 0
    if size > 0:
        complete_before_last = (size - 1) // 64
    for block_index in range(complete_before_last):
        load_block(data, offset + block_index * 64, 64, block)
        var block_flags = flags
        if block_index == 0:
            block_flags |= CHUNK_START
        compress(
            cv,
            block,
            chunk_counter,
            64,
            block_flags,
            compressed,
        )
        comptime W = simd_width_of[DType.float64]()
        var cv_pointer = cv.unsafe_ptr()
        var compressed_pointer = compressed.unsafe_ptr()
        var i = 0
        while i + W <= 8:
            cv_pointer.store(
                i, compressed_pointer.load[width=W](i)
            )
            i += W
        while i < 8:
            cv_pointer[i] = compressed_pointer[i]
            i += 1

    var last_size = size - complete_before_last * 64
    load_block(
        data,
        offset + complete_before_last * 64,
        last_size,
        output_block,
    )
    copy8(cv, output_cv)
    var output_flags = flags | CHUNK_END
    if complete_before_last == 0:
        output_flags |= CHUNK_START
    return (last_size, output_flags)


def chaining_value(
    input_cv: Array[UInt32, 8],
    block: Array[UInt32, 16],
    counter: UInt64,
    block_len: Int,
    flags: UInt32,
    mut result_cv: Array[UInt32, 8],
):
    var words = Array[UInt32, 16](fill=0)
    compress(input_cv, block, counter, block_len, flags, words)
    comptime W = simd_width_of[DType.float64]()
    var words_pointer = words.unsafe_ptr()
    var result_pointer = result_cv.unsafe_ptr()
    var i = 0
    while i + W <= 8:
        result_pointer.store(i, words_pointer.load[width=W](i))
        i += W
    while i < 8:
        result_pointer[i] = words_pointer[i]
        i += 1


def parent_cv(
    left: Array[UInt32, 8],
    right: Array[UInt32, 8],
    key: Array[UInt32, 8],
    flags: UInt32,
    mut result_cv: Array[UInt32, 8],
):
    var block = Array[UInt32, 16](fill=0)
    comptime W = simd_width_of[DType.float64]()
    var block_pointer = block.unsafe_ptr()
    var left_pointer = left.unsafe_ptr()
    var right_pointer = right.unsafe_ptr()
    var i = 0
    while i + W <= 8:
        block_pointer.store(i, left_pointer.load[width=W](i))
        block_pointer.store(
            i + 8, right_pointer.load[width=W](i)
        )
        i += W
    while i < 8:
        block_pointer[i] = left_pointer[i]
        block_pointer[i + 8] = right_pointer[i]
        i += 1
    chaining_value(key, block, 0, 64, flags | PARENT, result_cv)


def push_chunk_cv(
    chunk_index: Int,
    key: Array[UInt32, 8],
    flags: UInt32,
    mut stack: Array[UInt32, 432],
    stack_count: Int,
    mut current_cv: Array[UInt32, 8],
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var next_stack_count = stack_count
    var stack_pointer = stack.unsafe_ptr()
    var merged_cv = Array[UInt32, 8](fill=0)
    var left_cv = Array[UInt32, 8](fill=0)
    var total_chunks = chunk_index + 1
    while total_chunks % 2 == 0:
        next_stack_count -= 1
        var left_pointer = left_cv.unsafe_ptr()
        var stack_offset = next_stack_count * 8
        var i = 0
        while i + W <= 8:
            left_pointer.store(
                i,
                stack_pointer.load[width=W](stack_offset + i),
            )
            i += W
        while i < 8:
            left_pointer[i] = stack_pointer[stack_offset + i]
            i += 1
        parent_cv(left_cv, current_cv, key, flags, merged_cv)
        copy8(merged_cv, current_cv)
        total_chunks //= 2

    var current_pointer = current_cv.unsafe_ptr()
    var stack_offset = next_stack_count * 8
    var i = 0
    while i + W <= 8:
        stack_pointer.store(
            stack_offset + i,
            current_pointer.load[width=W](i),
        )
        i += W
    while i < 8:
        stack_pointer[stack_offset + i] = current_pointer[i]
        i += 1
    return next_stack_count + 1


def hash_impl(
    data: BPtr,
    size: Int,
    key: Array[UInt32, 8],
    flags: UInt32,
    seek: UInt64,
    destination: BPtr,
    destination_size: Int,
    max_workers: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var stack = Array[UInt32, 432](fill=0)
    var stack_pointer = stack.unsafe_ptr()
    var stack_count = 0
    var input_cv = Array[UInt32, 8](fill=0)
    var block = Array[UInt32, 16](fill=0)
    var current_cv = Array[UInt32, 8](fill=0)

    var chunk_count = 1
    if size > 0:
        chunk_count = (size + 1023) // 1024

    var worker_count = max_workers
    if worker_count < 0 or worker_count > 16:
        worker_count = 16
    if (
        size >= PARALLEL_THRESHOLD
        and chunk_count > 1
        and worker_count > 1
    ):
        var cvs = alloc[UInt32]((chunk_count - 1) * 8)

        @__parameter
        def hash_chunk(chunk_index: Int):
            var local_input_cv = Array[UInt32, 8](fill=0)
            var local_block = Array[UInt32, 16](fill=0)
            var local_cv = Array[UInt32, 8](fill=0)
            var metadata = chunk_output(
                data,
                chunk_index * 1024,
                1024,
                UInt64(chunk_index),
                key,
                flags,
                local_input_cv,
                local_block,
            )
            chaining_value(
                local_input_cv,
                local_block,
                UInt64(chunk_index),
                metadata[0],
                metadata[1],
                local_cv,
            )
            var local_pointer = local_cv.unsafe_ptr()
            var destination_offset = chunk_index * 8
            var i = 0
            while i + W <= 8:
                cvs.store(
                    destination_offset + i,
                    local_pointer.load[width=W](i),
                )
                i += W
            while i < 8:
                cvs[destination_offset + i] = local_pointer[i]
                i += 1

        parallelize[hash_chunk](chunk_count - 1, worker_count)
        var chunk_index = 0
        while chunk_index < chunk_count - 1:
            var current_pointer = current_cv.unsafe_ptr()
            var source_offset = chunk_index * 8
            var i = 0
            while i + W <= 8:
                current_pointer.store(
                    i, cvs.load[width=W](source_offset + i)
                )
                i += W
            while i < 8:
                current_pointer[i] = cvs[source_offset + i]
                i += 1
            stack_count = push_chunk_cv(
                chunk_index,
                key,
                flags,
                stack,
                stack_count,
                current_cv,
            )
            chunk_index += 1
        cvs.free()
    else:
        for chunk_index in range(chunk_count - 1):
            var metadata = chunk_output(
                data,
                chunk_index * 1024,
                1024,
                UInt64(chunk_index),
                key,
                flags,
                input_cv,
                block,
            )
            chaining_value(
                input_cv,
                block,
                UInt64(chunk_index),
                metadata[0],
                metadata[1],
                current_cv,
            )
            stack_count = push_chunk_cv(
                chunk_index,
                key,
                flags,
                stack,
                stack_count,
                current_cv,
            )

    var last_chunk = chunk_count - 1
    var last_size = size - last_chunk * 1024
    var root_metadata = chunk_output(
        data,
        last_chunk * 1024,
        last_size,
        UInt64(last_chunk),
        key,
        flags,
        input_cv,
        block,
    )
    var root_counter = UInt64(last_chunk)
    var root_block_len = root_metadata[0]
    var root_flags = root_metadata[1]

    while stack_count > 0:
        chaining_value(
            input_cv,
            block,
            root_counter,
            root_block_len,
            root_flags,
            current_cv,
        )
        stack_count -= 1
        var block_pointer = block.unsafe_ptr()
        var current_pointer = current_cv.unsafe_ptr()
        var key_pointer = key.unsafe_ptr()
        var input_pointer = input_cv.unsafe_ptr()
        var stack_offset = stack_count * 8
        var i = 0
        while i + W <= 8:
            block_pointer.store(
                i,
                stack_pointer.load[width=W](stack_offset + i),
            )
            block_pointer.store(
                i + 8, current_pointer.load[width=W](i)
            )
            input_pointer.store(i, key_pointer.load[width=W](i))
            i += W
        while i < 8:
            block_pointer[i] = stack_pointer[stack_offset + i]
            block_pointer[i + 8] = current_pointer[i]
            input_pointer[i] = key_pointer[i]
            i += 1
        root_counter = 0
        root_block_len = 64
        root_flags = flags | PARENT

    var output_words = Array[UInt32, 16](fill=0)
    var output_block_counter = seek // 64
    var skip = Int(seek % 64)
    var written = 0
    while written < destination_size:
        compress(
            input_cv,
            block,
            output_block_counter,
            root_block_len,
            root_flags | ROOT,
            output_words,
        )
        var output_bytes = output_words.unsafe_ptr().bitcast[UInt8]()
        var copy_size = min(64 - skip, destination_size - written)
        unsafe_memcpy(
            dest=destination + written,
            src=output_bytes + skip,
            count=copy_size,
        )
        written += copy_size
        skip = 0
        output_block_counter += 1


@export("mojo_blake3_hash")
def mojo_blake3_hash(
    data_addr: Int,
    size: Int,
    key_addr: Int,
    flags: Int,
    seek: UInt64,
    destination_addr: Int,
    destination_size: Int,
    max_workers: Int,
) abi("C") -> Int:
    initialize_runtime()
    if size < 0 or destination_size < 0:
        return -1
    if key_addr == 0:
        return -2
    if size > 0 and data_addr == 0:
        return -2
    if destination_size > 0 and destination_addr == 0:
        return -2
    if flags != 0 and flags != 16 and flags != 32 and flags != 64:
        return -3
    if max_workers != -1 and max_workers < 1:
        return -4
    # Pointer is non-nullable even when no bytes will be accessed.
    # Reuse the validated key pointer as a harmless sentinel for empty buffers.
    var safe_data_addr = data_addr
    var safe_destination_addr = destination_addr
    if safe_data_addr == 0:
        safe_data_addr = key_addr
    if safe_destination_addr == 0:
        safe_destination_addr = key_addr
    var data = BPtr(unsafe_from_address=safe_data_addr)
    var key_bytes = BPtr(unsafe_from_address=key_addr)
    var destination = BPtr(unsafe_from_address=safe_destination_addr)
    var key = Array[UInt32, 8](fill=0)
    for i in range(8):
        var offset = i * 4
        key[i] = (
            UInt32(key_bytes[offset])
            | (UInt32(key_bytes[offset + 1]) << 8)
            | (UInt32(key_bytes[offset + 2]) << 16)
            | (UInt32(key_bytes[offset + 3]) << 24)
        )
    hash_impl(
        data,
        size,
        key,
        UInt32(flags),
        seek,
        destination,
        destination_size,
        max_workers,
    )
    return 0
