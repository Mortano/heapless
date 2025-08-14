//! Implementations of `bytes` traits for `heapless` types.

use crate::{
    len_type::LenType,
    vec::{VecInner, VecStorage},
};
use bytes::{buf::UninitSlice, BufMut};

unsafe impl<S: VecStorage<u8> + ?Sized, LenT: LenType> BufMut for VecInner<u8, LenT, S> {
    #[inline]
    fn remaining_mut(&self) -> usize {
        self.capacity() - self.len()
    }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        let len = self.len();
        let pos = len + cnt;
        if pos >= self.capacity() {
            panic!("Advance out of range");
        }
        // SAFETY: Caller of `advance_mut` must ensure that the next `cnt` elements are initialized, which is the same
        //         safety guarantee that `set_len` has. In addition this can never go out of bounds due to the capacity
        //         check above
        self.set_len(pos);
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        let len = self.len();
        let ptr = self.as_mut_ptr();
        // SAFETY: The memory pointed to by `ptr` is valid as it comes from this buffer (so long as `VecStorage<u8>` is
        //         correctly implemented, which it is for the only two types that currently implement it)
        //         The underlying buffer has `capacity` capacity, so this is also valid
        //         Calling the omitted lifetime of `&mut self` `'a`, the memory region pointed to by `ptr` will be valid
        //         for at least `'a` since `as_mut_ptr` uses `self.borrow_mut` internally, which ties the lifetime of `'a`
        //         to the returned slice
        unsafe { &mut UninitSlice::from_raw_parts_mut(ptr, self.capacity())[len..] }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vec, VecView};
    use bytes::BufMut;

    #[test]
    #[should_panic]
    fn buf_mut_advance_mut_out_of_bounds() {
        let mut vec: Vec<u8, 8> = Vec::new();
        // SAFETY: Will cause a panic, as expected. In the current implementation, this panic happens before updating the length
        //         of the vector, so the safety guarantee is not violated
        unsafe { vec.advance_mut(9) };
    }

    #[test]
    fn buf_mut_remaining_mut() {
        let mut vec: Vec<u8, 8> = Vec::new();
        assert_eq!(vec.remaining_mut(), 8);
        vec.push(42).unwrap();
        assert_eq!(vec.remaining_mut(), 7);
    }

    #[test]
    fn buf_mut_chunk_mut() {
        let mut vec: Vec<u8, 8> = Vec::new();
        assert_eq!(vec.chunk_mut().len(), 8);
        // SAFETY: This is *not* safe as it violates the "memory must be initialized" guarantee. However we never access
        //         the falsely uninitialized memory, nor do we borrow it, so this is also *not* UB. In the context of this
        //         test, it is therefore fine to violate the guarantee
        unsafe { vec.advance_mut(1) };
        assert_eq!(vec.chunk_mut().len(), 7);
    }

    #[test]
    #[should_panic]
    fn buf_mut_advance_mut_out_of_bounds_view() {
        let vec: &mut VecView<u8, u8> = &mut Vec::<u8, 8, u8>::new();
        // SAFETY: Will cause a panic, as expected. In the current implementation, this panic happens before updating the length
        //         of the vector, so the safety guarantee is not violated
        unsafe { vec.advance_mut(9) };
    }

    #[test]
    fn buf_mut_remaining_mut_view() {
        let vec: &mut VecView<u8, u8> = &mut Vec::<u8, 8, u8>::new();
        assert_eq!(vec.remaining_mut(), 8);
        vec.push(42).unwrap();
        assert_eq!(vec.remaining_mut(), 7);
    }

    #[test]
    fn buf_mut_chunk_mut_view() {
        let vec: &mut VecView<u8, u8> = &mut Vec::<u8, 8, u8>::new();
        assert_eq!(vec.chunk_mut().len(), 8);
        // SAFETY: This is *not* safe as it violates the "memory must be initialized" guarantee. However we never access
        //         the falsely uninitialized memory, nor do we borrow it, so this is also *not* UB. In the context of this
        //         test, it is therefore fine to violate the guarantee
        unsafe { vec.advance_mut(1) };
        assert_eq!(vec.chunk_mut().len(), 7);
    }
}
