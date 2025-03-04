
#pragma once

#include <type_traits>
#include "initialise.hpp"

#if __cplusplus >= 202002L
namespace cms {
  using std::is_bounded_array;
  using std::is_unbounded_array;
}  // namespace cms
#else
#include <boost/type_traits/is_bounded_array.hpp>
#include <boost/type_traits/is_unbounded_array.hpp>
namespace cms {
  using boost::is_bounded_array;
  using boost::is_unbounded_array;
}  // namespace cms
#endif

namespace cms {
  template <class T>
  inline constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

  template <class T>
  inline constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;
}  // namespace cms

#include <alpaka/alpaka.hpp>

#include "AllocatorPolicy.hpp"
#include "CachedBufAlloc.hpp"
#include "alpakaConfig.hpp"
#include "alpakaDevices.hpp"

namespace clue {

  // for Extent, Dim1D, Idx
  using namespace alpaka_common;

  // type deduction helpers
  namespace detail {

    template <typename TDev, typename T>
    struct buffer_type {
      using type = alpaka::Buf<TDev, T, Dim0D, Idx>;
    };

    template <typename TDev, typename T>
    struct buffer_type<TDev, T[]> {
      using type = alpaka::Buf<TDev, T, Dim1D, Idx>;
    };

    template <typename TDev, typename T, int N>
    struct buffer_type<TDev, T[N]> {
      using type = alpaka::Buf<TDev, T, Dim1D, Idx>;
    };

    template <typename TDev, typename T>
    struct view_type {
      using type = alpaka::ViewPlainPtr<TDev, T, Dim0D, Idx>;
    };

    template <typename TDev, typename T>
    struct view_type<TDev, T[]> {
      using type = alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>;
    };

    template <typename TDev, typename T, int N>
    struct view_type<TDev, T[N]> {
      using type = alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>;
    };

  }  // namespace detail

  // scalar and 1-dimensional host buffers

  template <typename T>
  using host_buffer = typename detail::buffer_type<DevHost, T>::type;

  // non-cached, non-pinned, scalar and 1-dimensional host buffers

  template <typename T>
  std::enable_if_t<not std::is_array_v<T>, host_buffer<T>> make_host_buffer() {
    return alpaka::allocBuf<T, Idx>(host, Scalar{});
  }

  template <typename T>
  std::enable_if_t<cms::is_unbounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   host_buffer<T>>
  make_host_buffer(Extent extent) {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(host, Vec1D{extent});
  }

  template <typename T>
  std::enable_if_t<cms::is_bounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   host_buffer<T>>
  make_host_buffer() {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(host, Vec1D{std::extent_v<T>});
  }

  // potentially cached, pinned, scalar and 1-dimensional host buffers, associated to a work queue
  // the memory is pinned according to the device associated to the queue

  template <typename T, typename TQueue>
  std::enable_if_t<not std::is_array_v<T>, host_buffer<T>> make_host_buffer(
      TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<T, Idx>(host, queue, Scalar{});
    } else {
	  using Platform = alpaka::Platform<alpaka::Dev<TQueue>>;
      return alpaka::allocMappedBuf<T, Idx>(host, platform<Platform>(), Scalar{});
    }
  }

  template <typename T, typename TQueue>
  std::enable_if_t<cms::is_unbounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   host_buffer<T>>
  make_host_buffer(TQueue const& queue, Extent extent) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(host, queue, Vec1D{extent});
    } else {
      using Platform = alpaka::Platform<alpaka::Dev<TQueue>>;
      return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(
          host, platform<Platform>(), Vec1D{extent});
    }
  }

  template <typename T, typename TQueue>
  std::enable_if_t<cms::is_bounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   host_buffer<T>>
  make_host_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(
          host, queue, Vec1D{std::extent_v<T>});
    } else {
	  using Platform = alpaka::Platform<alpaka::Dev<TQueue>>;
      return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(
          host, platform<Platform>(), Vec1D{std::extent_v<T>});
    }
  }

  // scalar and 1-dimensional host views

  template <typename T>
  using host_view = typename detail::view_type<DevHost, T>::type;

  template <typename T>
  std::enable_if_t<not std::is_array_v<T>, host_view<T>> make_host_view(T& data) {
    return alpaka::ViewPlainPtr<DevHost, T, Dim0D, Idx>(&data, host, Scalar{});
  }

  template <typename T>
  host_view<T[]> make_host_view(T* data, Extent extent) {
    return alpaka::ViewPlainPtr<DevHost, T, Dim1D, Idx>(data, host, Vec1D{extent});
  }

  template <typename T>
  std::enable_if_t<cms::is_unbounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   host_view<T>>
  make_host_view(T& data, Extent extent) {
    return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(
        data, host, Vec1D{extent});
  }

  template <typename T>
  std::enable_if_t<cms::is_bounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   host_view<T>>
  make_host_view(T& data) {
    return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(
        data, host, Vec1D{std::extent_v<T>});
  }

  // scalar and 1-dimensional device buffers

  template <typename TDev, typename T>
  using device_buffer = typename detail::buffer_type<TDev, T>::type;

  template <typename T, typename TQueue>
  std::enable_if_t<not std::is_array_v<T>, device_buffer<alpaka::Dev<TQueue>, T>>
  make_device_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<T, Idx>(alpaka::getDev(queue), queue, Scalar{});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> ==
                  AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<T, Idx>(queue, Scalar{});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<T, Idx>(alpaka::getDev(queue), Scalar{});
    }
  }

  template <typename T, typename TQueue>
  std::enable_if_t<cms::is_unbounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   device_buffer<alpaka::Dev<TQueue>, T>>
  make_device_buffer(TQueue const& queue, Extent extent) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(
          alpaka::getDev(queue), queue, Vec1D{extent});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> ==
                  AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue, Vec1D{extent});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue),
                                                            Vec1D{extent});
    }
  }

  template <typename T, typename TQueue>
  std::enable_if_t<cms::is_bounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   device_buffer<alpaka::Dev<TQueue>, T>>
  make_device_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(
          alpaka::getDev(queue), queue, Vec1D{std::extent_v<T>});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> ==
                  AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue,
                                                                 Vec1D{std::extent_v<T>});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue),
                                                            Vec1D{std::extent_v<T>});
    }
  }

  // scalar and 1-dimensional device views

  template <typename TDev, typename T>
  using device_view = typename detail::view_type<TDev, T>::type;

  template <typename T, typename TDev>
  std::enable_if_t<not std::is_array_v<T>, device_view<TDev, T>> make_device_view(
      TDev const& device, T& data) {
    return alpaka::ViewPlainPtr<TDev, T, Dim0D, Idx>(&data, device, Scalar{});
  }

  template <typename T, typename TDev>
  device_view<TDev, T[]> make_device_view(TDev const& device, T* data, Extent extent) {
    return alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>(data, device, Vec1D{extent});
  }

  template <typename T, typename TDev>
  std::enable_if_t<cms::is_unbounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   device_view<TDev, T>>
  make_device_view(TDev const& device, T& data, Extent extent) {
    return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(
        data, device, Vec1D{extent});
  }

  template <typename T, typename TDev>
  std::enable_if_t<cms::is_bounded_array_v<T> and
                       not std::is_array_v<std::remove_extent_t<T>>,
                   device_view<TDev, T>>
  make_device_view(TDev const& device, T& data) {
    return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(
        data, device, Vec1D{std::extent_v<T>});
  }

}  // namespace clue
