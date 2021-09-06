pub struct MappableStorageBuffer<T> {
    pub(crate) inner: wgpu::Buffer,
    pub(crate) size: u32,
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> UploadableBuffer for MappableStorageBuffer<T> {
    type Source = [u32];

    fn map_source_to_bytes(source: &Self::Source) -> &[u8] {
        bytemuck::cast_slice(source)
    }

    fn new_from_buffer(buffer: wgpu::Buffer, source: &Self::Source) -> Self {
        Self {
            inner: buffer,
            size: source.len() as u32,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T> MappableBuffer<'a> for MappableStorageBuffer<T> {
    type Mapped = MappedStorageBuffer<'a, T>;

    fn slice(&self) -> wgpu::BufferSlice {
        self.inner.slice(..)
    }

    fn from_mapped_slice(slice: wgpu::BufferSlice<'a>) -> Self::Mapped {
        Self::Mapped {
            view: slice.get_mapped_range(),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct MappedStorageBuffer<'a, T> {
    view: wgpu::BufferView<'a>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: bytemuck::Pod> std::ops::Deref for MappedStorageBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(&self.view)
    }
}

pub struct MappableSingleValueBuffer<T> {
    pub(crate) inner: wgpu::Buffer,
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> UploadableBuffer for MappableSingleValueBuffer<T> {
    type Source = u32;

    fn map_source_to_bytes(source: &Self::Source) -> &[u8] {
        bytemuck::bytes_of(source)
    }

    fn new_from_buffer(buffer: wgpu::Buffer, _source: &Self::Source) -> Self {
        Self {
            inner: buffer,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T> MappableBuffer<'a> for MappableSingleValueBuffer<T> {
    type Mapped = MappedSingleValueBuffer<'a, T>;

    fn slice(&self) -> wgpu::BufferSlice {
        self.inner.slice(..)
    }

    fn from_mapped_slice(slice: wgpu::BufferSlice<'a>) -> Self::Mapped {
        Self::Mapped {
            view: slice.get_mapped_range(),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct MappedSingleValueBuffer<'a, T> {
    view: wgpu::BufferView<'a>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: bytemuck::Pod> std::ops::Deref for MappedSingleValueBuffer<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        bytemuck::from_bytes(&self.view)
    }
}

pub trait UploadableBuffer {
    type Source: ?Sized;

    fn map_source_to_bytes(source: &Self::Source) -> &[u8];

    fn new_from_buffer(buffer: wgpu::Buffer, source: &Self::Source) -> Self;
}

pub trait MappableBuffer<'a> {
    type Mapped;

    fn slice(&self) -> wgpu::BufferSlice;

    fn from_mapped_slice(slice: wgpu::BufferSlice<'a>) -> Self::Mapped;
}
